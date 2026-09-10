#!/usr/bin/env python3
"""Produce the full report: tables, figures, sensitivities.

    python scripts/run_report.py                      # synthetic fixture data
    python scripts/run_report.py --bars data_cache/bars.parquet

Writes markdown tables to reports/results.md and PNGs to reports/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from backtester import data, report, signals
from backtester.costs import CostModel
from backtester.engine import run_backtest
from backtester.leakguard import lag_sensitivity
from backtester.walkforward import refit_sensitivity, walk_forward

OUT = Path(__file__).resolve().parent.parent / "reports"

STRATEGIES = {
    "buy_and_hold": signals.BuyAndHold,
    "momentum": lambda: signals.Momentum(lookback=60),
    "mean_reversion": lambda: signals.MeanReversion(window=20),
    "vol_filtered_momentum": lambda: signals.VolFilteredMomentum(lookback=60),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", default=None)
    ap.add_argument("--n-days", type=int, default=3000)
    ap.add_argument("--train-bars", type=int, default=750)
    ap.add_argument("--test-bars", type=int, default=125)
    ap.add_argument("--half-spread-bps", type=float, default=1.0)
    ap.add_argument("--commission-bps", type=float, default=0.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    bars = (
        data.load(args.bars)
        if args.bars
        else data.synthetic_panel(n_days=args.n_days)
    )
    costs = CostModel(args.half_spread_bps, args.commission_bps)

    source = args.bars or f"synthetic (seed 7, {args.n_days} bars)"
    lines: list[str] = [
        "# Results",
        "",
        f"- data: `{source}`",
        f"- span: {bars['date'].min()} to {bars['date'].max()}, "
        f"{bars['ticker'].n_unique()} tickers, {bars['date'].n_unique():,} bars",
        f"- costs: {costs.describe()}",
        "- execution: decide at close t-1, fill at open t, mark out at open t+1",
        "- risk-free rate assumed 0.0 in all Sharpe figures",
        "",
    ]

    # ---- 1. Full-sample, all strategies -----------------------------------
    full = {name: run_backtest(bars, f(), costs=costs) for name, f in STRATEGIES.items()}
    lines += [
        "## Full-sample summary (in-sample; NOT the result)",
        "",
        "Shown for completeness and as a sanity check on the plumbing. The",
        "number to quote is the walk-forward one below.",
        "",
        report.markdown_table(report.summary_table(full)),
        "",
    ]

    # ---- 2. Walk-forward ---------------------------------------------------
    lines += ["## Walk-forward, out-of-sample only", ""]
    wf_results = {}
    for name, factory in STRATEGIES.items():
        wf = walk_forward(
            bars, factory, train_bars=args.train_bars, test_bars=args.test_bars, costs=costs
        )
        wf_results[name] = wf
    wf_tbl = pl.DataFrame(
        [
            {
                "strategy": n,
                "folds": len(w.folds),
                "oos_sharpe": w.metrics.sharpe,
                "oos_gross_sharpe": w.metrics.gross_sharpe,
                "oos_cagr": w.metrics.cagr,
                "oos_max_dd": w.metrics.max_drawdown,
                "turnover": w.metrics.turnover,
                "exposure": w.metrics.exposure,
            }
            for n, w in wf_results.items()
        ]
    )
    lines += [report.markdown_table(wf_tbl), ""]

    best = max(wf_results, key=lambda k: wf_results[k].metrics.sharpe)
    lines += [
        f"Per-fold detail for `{best}` (the spread across folds matters more than the mean):",
        "",
        report.markdown_table(wf_results[best].fold_table()),
        "",
    ]

    # ---- 3. Cost sensitivity ----------------------------------------------
    lines += ["## Cost sensitivity", ""]
    sens = {}
    for name in ("momentum", "mean_reversion", "vol_filtered_momentum"):
        s = report.cost_sensitivity(bars, STRATEGIES[name])
        sens[name] = s
        be = report.breakeven_cost_bps(s)
        lines += [
            f"### {name}",
            "",
            report.markdown_table(s),
            "",
            f"Break-even all-in cost: **{be:.1f} bps**"
            if be == be
            else "Break-even all-in cost: beyond the tested grid.",
            "",
        ]
    report.plot_cost_sensitivity(sens, OUT / "cost_sensitivity.png")

    # ---- 4. Latency sensitivity (the leak diagnostic) ----------------------
    lines += [
        "## Latency sensitivity",
        "",
        "Extra bars of delay ON TOP of the mandatory one. A signal whose Sharpe",
        "falls off a cliff between 0 and 1 was reading something close to its own",
        "fill price. `peek_ahead_momentum` is included as the positive control.",
        "",
    ]
    lags = {
        "momentum": lag_sensitivity(bars, signals.Momentum(60), lags=(0, 1, 2, 3, 5)),
        "mean_reversion": lag_sensitivity(bars, signals.MeanReversion(20), lags=(0, 1, 2, 3, 5)),
        "peek_ahead_momentum": lag_sensitivity(
            bars, signals.PeekAheadMomentum(5), lags=(0, 1, 2, 3, 5), audit=False
        ),
    }
    for name, df in lags.items():
        lines += [f"### {name}", "", report.markdown_table(df), ""]
    report.plot_lag_sensitivity(lags, OUT / "lag_sensitivity.png")

    # ---- 5. Refit-cadence sensitivity --------------------------------------
    rs = refit_sensitivity(
        bars, STRATEGIES[best], test_bars_grid=(21, 63, 125, 252),
        train_bars=args.train_bars, costs=costs,
    )
    lines += [
        "## Refit-cadence sensitivity",
        "",
        f"Out-of-sample Sharpe for `{best}` as the refit interval changes. Flat is",
        "good news; a single lucky cadence is not a result.",
        "",
        report.markdown_table(rs),
        "",
    ]

    # ---- 6. Figures --------------------------------------------------------
    report.plot_equity_and_drawdown(
        _as_result(wf_results[best]),
        OUT / "equity.png",
        title=f"{best}: walk-forward out-of-sample equity, net of costs",
    )
    lines += [
        "## Figures",
        "",
        "![equity](equity.png)",
        "",
        "![cost sensitivity](cost_sensitivity.png)",
        "",
        "![lag sensitivity](lag_sensitivity.png)",
        "",
    ]

    (OUT / "results.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'results.md'} and 3 figures")


def _as_result(wf):
    """Adapt a WalkForwardResult to the shape the plotting helper expects."""
    from backtester.engine import BacktestResult

    return BacktestResult(panel=wf.portfolio, portfolio=wf.portfolio,
                          metrics=wf.metrics, config=wf.config)


if __name__ == "__main__":
    main()
