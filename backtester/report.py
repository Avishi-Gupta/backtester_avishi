"""Summary tables, figures and sensitivity grids.

Chart conventions:

* One y-axis per panel. Equity and drawdown are plotted as two stacked panels
  sharing an x-axis rather than on twin axes.
* Net is the primary series; gross is drawn thinner and lighter.
* Series are named in the legend and direct-labelled at the right edge, so
  identity does not depend on colour alone.
* Recessive grid and axes; no value labels on individual points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import polars as pl

from backtester.costs import CostModel
from backtester.engine import BacktestResult, run_backtest
from backtester.metrics import Metrics, metrics_table

# Colours chosen for separability under common colour-vision deficiencies.
NET = "#2a78d6"      # blue   - net series
GROSS = "#eb6834"    # orange - gross series
LOSS = "#e34948"     # red    - drawdown
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def summary_table(results: dict[str, BacktestResult]) -> pl.DataFrame:
    """One row per strategy, in a fixed column order.

    Net and gross Sharpe are placed next to turnover so a high Sharpe with high
    turnover is visible as a cost problem.
    """
    tbl = metrics_table({k: v.metrics for k, v in results.items()})
    cols = [
        "strategy", "sharpe", "gross_sharpe", "cagr", "ann_vol",
        "max_drawdown", "calmar", "turnover", "exposure", "hit_rate",
        "n_trades", "avg_cost_bps", "n_periods",
    ]
    return tbl.select([c for c in cols if c in tbl.columns])


def cost_sensitivity(
    bars: pl.DataFrame,
    signal_factory: Callable[[], Any],
    base: CostModel | None = None,
    all_in_bps: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    **kwargs: Any,
) -> pl.DataFrame:
    """Sharpe as a function of all-in transaction cost."""
    rows = []
    for bps in all_in_bps:
        model = CostModel(half_spread_bps=bps, commission_bps=0.0)
        res = run_backtest(bars, signal_factory(), costs=model, **kwargs)
        m = res.metrics
        rows.append(
            {
                "all_in_bps": bps,
                "sharpe_net": m.sharpe,
                "cagr": m.cagr,
                "max_drawdown": m.max_drawdown,
                "turnover": m.turnover,
                "annual_cost_drag": m.turnover * 252 * bps * 1e-4,
            }
        )
    _ = base
    return pl.DataFrame(rows)


def walkforward_cost_sensitivity(
    bars: pl.DataFrame,
    signal_factory: Callable[[], Any],
    all_in_bps: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    train_bars: int = 750,
    test_bars: int = 125,
    **kwargs: Any,
) -> pl.DataFrame:
    """Cost sensitivity measured out-of-sample, one walk-forward run per cost level.

    ``cost_sensitivity`` above runs on the full sample, which is cheaper but is
    an in-sample figure. This version re-runs the walk-forward at each cost
    level, so the break-even it feeds is an out-of-sample number.
    """
    from backtester.walkforward import walk_forward

    rows = []
    for bps in all_in_bps:
        model = CostModel(half_spread_bps=bps, commission_bps=0.0)
        wf = walk_forward(
            bars, signal_factory, train_bars=train_bars, test_bars=test_bars,
            costs=model, **kwargs,
        )
        m = wf.metrics
        rows.append(
            {
                "all_in_bps": bps,
                "sharpe_net": m.sharpe,
                "cagr": m.cagr,
                "max_drawdown": m.max_drawdown,
                "turnover": m.turnover,
                "n_folds": len(wf.folds),
            }
        )
    return pl.DataFrame(rows)


def breakeven_cost_bps(sens: pl.DataFrame) -> float:
    """Linear interpolation of the cost level at which net Sharpe hits zero.

    Returns NaN if the strategy does not cross zero inside the tested grid,
    rather than extrapolating beyond it.
    """
    x = sens["all_in_bps"].to_numpy()
    y = sens["sharpe_net"].to_numpy()
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            t = y[i] / (y[i] - y[i + 1])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def _direct_labels(ax, items, min_gap_px: float = 13.0) -> None:
    """Right-edge series labels, pushed apart so they never overprint.

    Labels are placed at each series' final value, then nudged vertically in
    display space until adjacent labels clear each other.
    """
    if not items:
        return
    pts = sorted(((float(y), x, text, colour) for x, y, text, colour in items))
    ypix = [ax.transData.transform((0, y))[1] for y, *_ in pts]
    for i in range(1, len(ypix)):
        if ypix[i] - ypix[i - 1] < min_gap_px:
            ypix[i] = ypix[i - 1] + min_gap_px
    for (y, x, text, colour), target in zip(pts, ypix):
        dy = target - ax.transData.transform((0, y))[1]
        ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                    color=colour, fontsize=9, va="center", annotation_clip=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_equity_and_drawdown(
    result: BacktestResult,
    path: str | Path,
    title: str = "Out-of-sample equity, net of costs",
) -> Path:
    """Two stacked panels: log equity on top, underwater plot beneath.

    Equity uses a log scale so that proportional moves are comparable across
    the sample rather than dominated by the later years.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter

    p = result.portfolio
    dates = p["date"].to_list()
    net = p["equity"].to_numpy()
    gross = p["gross_equity"].to_numpy()
    dd = p["drawdown"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    ax1.plot(dates, gross, color=GROSS, linewidth=1.2, alpha=0.85, label="Gross", zorder=2)
    ax1.plot(dates, net, color=NET, linewidth=2.0, label="Net of costs", zorder=3)
    ax1.set_yscale("log")
    # A log axis defaults to decade ticks, which on a 1x-4x equity curve leaves
    # a single labelled gridline. Pick nice multiplicative steps inside the
    # actual range instead.
    lo, hi = float(min(net.min(), gross.min())), float(max(net.max(), gross.max()))
    ladder = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
    ticks = [v for v in ladder if lo * 0.98 <= v <= hi * 1.02]
    if len(ticks) >= 2:
        ax1.set_yticks(ticks)
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_ylabel("Equity (log, start = 1)", color=INK_2, fontsize=9)
    ax1.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    _style(ax1)

    # Net and gross end close together, so offset the labels to avoid overlap.
    span = ax1.transData.transform((0, max(net[-1], gross[-1])))[1] - ax1.transData.transform(
        (0, min(net[-1], gross[-1]))
    )[1]
    dy = 0.0 if span > 11 else (11 - span) / 2
    ax1.annotate(f"Net {net[-1]:.2f}x", (dates[-1], net[-1]),
                 xytext=(6, -dy if net[-1] < gross[-1] else dy),
                 textcoords="offset points", color=NET, fontsize=9, va="center")
    ax1.annotate(f"Gross {gross[-1]:.2f}x", (dates[-1], gross[-1]),
                 xytext=(6, dy if gross[-1] > net[-1] else -dy),
                 textcoords="offset points", color=INK_2, fontsize=9, va="center")
    leg = ax1.legend(frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax2.fill_between(dates, dd, 0.0, color=LOSS, alpha=0.22, linewidth=0, zorder=2)
    ax2.plot(dates, dd, color=LOSS, linewidth=1.2, zorder=3)
    ax2.set_ylabel("Drawdown", color=INK_2, fontsize=9)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax2)

    # Label the trough, flipping inward near the right edge to avoid clipping.
    worst = int(np.argmin(dd))
    late = worst > 0.75 * len(dd)
    ax2.annotate(
        f"worst {dd[worst]:.1%}", (dates[worst], dd[worst]),
        xytext=(-6 if late else 6, 6), textcoords="offset points",
        color=INK, fontsize=9, ha="right" if late else "left",
    )

    fig.subplots_adjust(right=0.85, left=0.09, top=0.9, bottom=0.09)
    p_out = Path(path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p_out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p_out


def plot_cost_sensitivity(
    sens_by_strategy: dict[str, pl.DataFrame],
    path: str | Path,
    title: str = "Net Sharpe vs transaction cost",
) -> Path:
    """Net Sharpe against cost, one line per strategy. Capped at three."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = [NET, GROSS, "#1baf7a"]
    labels: list[tuple] = []
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)

    for i, (name, df) in enumerate(list(sens_by_strategy.items())[:3]):
        x = df["all_in_bps"].to_numpy()
        y = df["sharpe_net"].to_numpy()
        c = colours[i % len(colours)]
        ax.plot(x, y, color=c, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        labels.append((x[-1], y[-1], name, c))

    ax.axhline(0.0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlabel("All-in cost per unit traded (bps)", color=INK_2, fontsize=9)
    ax.set_ylabel("Net Sharpe", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    # Legend below the axes, since direct labels occupy the right edge.
    leg = ax.legend(frameon=False, fontsize=9, ncol=3,
                    loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for t in leg.get_texts():
        t.set_color(INK_2)
    _direct_labels(ax, labels)

    fig.subplots_adjust(right=0.78, left=0.09, top=0.88, bottom=0.26)
    p_out = Path(path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p_out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p_out


def plot_lag_sensitivity(
    lag_by_strategy: dict[str, pl.DataFrame],
    path: str | Path,
    title: str = "Sharpe decay with execution latency",
) -> Path:
    """Sharpe against added execution delay, per strategy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = [NET, GROSS, "#1baf7a"]
    labels: list[tuple] = []
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)

    for i, (name, df) in enumerate(list(lag_by_strategy.items())[:3]):
        x = df["extra_lag_bars"].to_numpy()
        y = df["sharpe_net"].to_numpy()
        ax.plot(x, y, color=colours[i % 3], linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        labels.append((x[-1], y[-1], name, colours[i % 3]))

    ax.axhline(0.0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlabel("Extra bars of delay on top of the mandatory t+1", color=INK_2, fontsize=9)
    ax.set_ylabel("Net Sharpe", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    _style(ax)
    # Legend below the axes, since direct labels occupy the right edge.
    leg = ax.legend(frameon=False, fontsize=9, ncol=3,
                    loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for t in leg.get_texts():
        t.set_color(INK_2)
    _direct_labels(ax, labels)

    fig.subplots_adjust(right=0.78, left=0.09, top=0.88, bottom=0.26)
    p_out = Path(path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p_out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p_out


def markdown_table(df: pl.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a Polars frame as a GitHub-flavoured markdown table."""
    cols = df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("n/a" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
