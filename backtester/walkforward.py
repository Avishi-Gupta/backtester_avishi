"""Rolling train/test evaluation.

A single train/test split is fine exactly once. The moment you look at the test
score and change something, the test set has become a training set, and after a
dozen such rounds your "out-of-sample" Sharpe is a measure of how many times
you re-ran the notebook.

Walk-forward does not solve that (nothing does, short of a locked-away holdout
and a pre-registered hypothesis), but it makes it much more expensive to fool
yourself: every fold is scored on data the model had not seen when it was fit,
and the result you quote is the concatenation of those out-of-sample segments
and nothing else. The in-sample curve is not a result and is not returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import polars as pl

from backtester.costs import CostModel
from backtester.engine import BacktestResult, ExecutionRule, run_backtest
from backtester.metrics import Metrics, drawdown_series, equity_curve, summarise

Mode = Literal["rolling", "expanding"]


@dataclass
class Fold:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: Metrics


@dataclass
class WalkForwardResult:
    portfolio: pl.DataFrame   # concatenated OUT-OF-SAMPLE bars only
    metrics: Metrics          # computed on that concatenation
    folds: list[Fold]
    config: dict[str, Any]

    def fold_table(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "fold": f.index,
                    "train_bars": f.train_end - f.train_start,
                    "test_bars": f.test_end - f.test_start,
                    "sharpe": f.metrics.sharpe,
                    "max_drawdown": f.metrics.max_drawdown,
                    "turnover": f.metrics.turnover,
                }
                for f in self.folds
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        m = self.metrics
        return (
            f"<WalkForward folds={len(self.folds)} oos_sharpe={m.sharpe:.2f} "
            f"maxDD={m.max_drawdown:.1%}>"
        )


def walk_forward(
    bars: pl.DataFrame,
    signal_factory: Callable[[], Any],
    train_bars: int = 750,
    test_bars: int = 125,
    mode: Mode = "rolling",
    costs: CostModel | None = None,
    execution: ExecutionRule = "next_open",
    periods_per_year: int = 252,
    audit: bool = True,
    **kwargs: Any,
) -> WalkForwardResult:
    """Fit, evaluate forward, slide, repeat.

    ``signal_factory`` is a callable returning a FRESH signal, not a signal
    instance. A single instance carried across folds would keep whatever state
    it learned in fold 1 when it is scored on fold 2, which is leakage wearing a
    different hat.

    ``test_bars`` doubles as the refit interval. Vary it and report the spread:
    if the result only survives at one refit cadence, it is a coincidence.
    """
    dates = bars["date"].unique().sort()
    n = dates.len()
    if train_bars + test_bars > n:
        raise ValueError(
            f"need at least {train_bars + test_bars} distinct dates, panel has {n}"
        )

    folds: list[Fold] = []
    segments: list[pl.DataFrame] = []

    test_start = train_bars
    fold_i = 0
    while test_start + test_bars <= n:
        test_end = test_start + test_bars
        train_start = 0 if mode == "expanding" else max(0, test_start - train_bars)

        train_dates = dates[train_start:test_start]
        # The signal is fit on the training window only. It is then RUN over
        # history up to test_end, because it needs warmup bars to produce a
        # position on the first test day -- but only the test rows are kept.
        train_panel = bars.filter(pl.col("date").is_in(train_dates.implode()))
        run_dates = dates[:test_end]
        run_panel = bars.filter(pl.col("date").is_in(run_dates.implode()))

        sig = signal_factory()
        if hasattr(sig, "fit"):
            sig.fit(train_panel)

        res: BacktestResult = run_backtest(
            run_panel,
            sig,
            costs=costs,
            execution=execution,
            periods_per_year=periods_per_year,
            audit=audit,
            **kwargs,
        )

        oos_dates = dates[test_start:test_end]
        seg = res.portfolio.filter(pl.col("date").is_in(oos_dates.implode())).sort("date")
        segments.append(seg)

        folds.append(
            Fold(
                index=fold_i,
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                metrics=summarise(
                    seg["net_ret"].to_numpy(),
                    seg["gross_ret"].to_numpy(),
                    seg["position"].to_numpy(),
                    seg["cost"].to_numpy(),
                    periods_per_year=periods_per_year,
                ),
            )
        )
        fold_i += 1
        test_start = test_end

    stitched = pl.concat(segments).sort("date").drop(["equity", "drawdown", "gross_equity"])
    eq = equity_curve(stitched["net_ret"].to_numpy())
    stitched = stitched.with_columns(
        pl.Series("equity", eq),
        pl.Series("drawdown", drawdown_series(eq)),
        pl.Series("gross_equity", equity_curve(stitched["gross_ret"].to_numpy())),
    )

    metrics = summarise(
        stitched["net_ret"].to_numpy(),
        stitched["gross_ret"].to_numpy(),
        stitched["position"].to_numpy(),
        stitched["cost"].to_numpy(),
        periods_per_year=periods_per_year,
    )

    return WalkForwardResult(
        portfolio=stitched,
        metrics=metrics,
        folds=folds,
        config={
            "train_bars": train_bars,
            "test_bars": test_bars,
            "mode": mode,
            "n_folds": len(folds),
            "execution": execution,
        },
    )


def refit_sensitivity(
    bars: pl.DataFrame,
    signal_factory: Callable[[], Any],
    test_bars_grid: tuple[int, ...] = (21, 63, 125, 252),
    **kwargs: Any,
) -> pl.DataFrame:
    """Out-of-sample Sharpe as a function of refit cadence.

    Flat is good news: the result does not depend on a lucky choice of
    hyper-parameter you never thought of as one.
    """
    rows = []
    for tb in test_bars_grid:
        try:
            r = walk_forward(bars, signal_factory, test_bars=tb, **kwargs)
        except ValueError:
            continue
        rows.append(
            {
                "test_bars": tb,
                "n_folds": len(r.folds),
                "oos_sharpe": r.metrics.sharpe,
                "oos_max_dd": r.metrics.max_drawdown,
                "turnover": r.metrics.turnover,
            }
        )
    return pl.DataFrame(rows)
