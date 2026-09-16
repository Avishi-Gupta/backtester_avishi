"""Position-to-PnL engine.

A position decided from bar ``t`` data is executed at bar ``t+1``. The shift is
applied here, before positions are multiplied by returns, so signal code cannot
change or omit it.

Timeline for the default ``execution="next_open"``:

    close of bar t-1   signal sees rows 0..t-1 and returns a target position
    open  of bar t     the engine fills that target
    open  of bar t+1   the position is marked out; ret_t = open_{t+1}/open_t - 1

So ``held_t = raw_{t-1}``: one bar of latency, and the fill price is not a price
the signal observed.

``execution="close_to_close"`` earns ``close_t/close_{t-1} - 1`` during bar t,
which requires a fill at ``close_{t-1}``. Since a decision taken at that close
cannot also be filled at it, this mode carries two bars of latency
(``held_t = raw_{t-2}``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from backtester import data as data_mod
from backtester.costs import CostModel
from backtester.metrics import Metrics, drawdown_series, equity_curve, summarise
from backtester.signal import PositionError, validate_positions

ExecutionRule = Literal["next_open", "close_to_close"]

#: Bars of latency each execution rule imposes. ``extra_lag`` adds to this.
BASE_LAG: dict[str, int] = {"next_open": 1, "close_to_close": 2}


@dataclass
class BacktestResult:
    """Output of a single backtest run, with the config used to produce it."""

    panel: pl.DataFrame          # per (date, ticker) detail
    portfolio: pl.DataFrame      # per date, aggregated
    metrics: Metrics
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def equity(self) -> np.ndarray:
        return self.portfolio["equity"].to_numpy()

    @property
    def net_returns(self) -> np.ndarray:
        return self.portfolio["net_ret"].to_numpy()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        m = self.metrics
        return (
            f"<BacktestResult n={m.n_periods} sharpe={m.sharpe:.2f} "
            f"(gross {m.gross_sharpe:.2f}) maxDD={m.max_drawdown:.1%} "
            f"turnover={m.turnover:.3f}>"
        )


def _positions_rowwise(signal: Any, frame: pl.DataFrame) -> np.ndarray:
    """Call ``predict`` once per bar, passing only rows 0..t.

    Polars slices are zero-copy views, so memory is O(n) rather than O(n^2).
    Runtime is one Python call per bar.
    """
    n = frame.height
    out = np.empty(n, dtype=float)
    for t in range(n):
        out[t] = signal.predict(frame.slice(0, t + 1))
    return out


def _target_positions(
    signal: Any,
    frame: pl.DataFrame,
    audit: bool,
) -> np.ndarray:
    """Get raw target positions from either signal interface."""
    if hasattr(signal, "predict_all"):
        if audit:
            # Imported here to avoid a circular import at module load.
            from backtester.leakguard import audit_vector_signal

            audit_vector_signal(signal, frame)
        series = signal.predict_all(frame)
        if len(series) != frame.height:
            raise PositionError(
                f"predict_all returned {len(series)} positions for {frame.height} bars"
            )
        return validate_positions(pl.Series(series)).to_numpy()
    if hasattr(signal, "predict"):
        raw = _positions_rowwise(signal, frame)
        return validate_positions(pl.Series(raw)).to_numpy()
    raise TypeError(
        "signal must implement either predict(history) -> float or "
        "predict_all(frame) -> Series"
    )


def run_backtest(
    bars: pl.DataFrame,
    signal: Any,
    costs: CostModel | None = None,
    execution: ExecutionRule = "next_open",
    extra_lag: int = 0,
    periods_per_year: int = 252,
    rf: float = 0.0,
    audit: bool = True,
) -> BacktestResult:
    """Run one signal over one panel of bars.

    Parameters
    ----------
    extra_lag:
        Additional bars of delay on top of the execution lag, used to measure
        how quickly a signal's Sharpe decays with latency.
    audit:
        Whether to run the future-perturbation audit on a vectorised signal.
        The choice is recorded in ``result.config["audited"]``.
    """
    if extra_lag < 0:
        raise ValueError("extra_lag must be >= 0")
    costs = costs if costs is not None else CostModel()

    data_mod.validate_bars(bars)
    panel = data_mod.to_returns(bars.sort(["ticker", "date"]), execution=execution)

    lag = BASE_LAG[execution] + extra_lag
    tickers = panel["ticker"].unique(maintain_order=True).to_list()

    parts: list[pl.DataFrame] = []
    for tk in tickers:
        sub = panel.filter(pl.col("ticker") == tk)
        raw = _target_positions(signal, sub, audit=audit)
        parts.append(sub.with_columns(pl.Series("raw_position", raw)))
    panel = pl.concat(parts)

    dollar_volume = pl.col("close") * pl.col("volume")

    panel = panel.with_columns(
        # Execution lag: a position decided from bar t is held from bar t+1.
        pl.col("raw_position").shift(lag).over("ticker").fill_null(0.0).alias("position")
    ).with_columns(
        # An untradable bar accrues no return; the position is carried.
        pl.when(pl.col("ret").is_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("ret"))
        .alias("ret"),
    ).with_columns(
        (pl.col("position") - pl.col("position").shift(1).over("ticker").fill_null(0.0))
        .abs()
        .alias("traded")
    ).with_columns(
        costs.per_bar_cost(pl.col("traded"), dollar_volume).alias("cost")
    ).with_columns(
        (pl.col("position") * pl.col("ret")).alias("gross_ret"),
    ).with_columns(
        (pl.col("gross_ret") - pl.col("cost")).alias("net_ret")
    )

    # Equal weight across names, keeping gross exposure <= 1 regardless of
    # panel size. Other weighting schemes are portfolio construction, not
    # execution, and are out of scope for the engine.
    n = max(len(tickers), 1)
    portfolio = (
        panel.group_by("date")
        .agg(
            (pl.col("position").sum() / n).alias("position"),
            (pl.col("gross_ret").sum() / n).alias("gross_ret"),
            (pl.col("cost").sum() / n).alias("cost"),
            (pl.col("net_ret").sum() / n).alias("net_ret"),
            (pl.col("traded").sum() / n).alias("traded"),
        )
        .sort("date")
    )

    eq = equity_curve(portfolio["net_ret"].to_numpy())
    portfolio = portfolio.with_columns(
        pl.Series("equity", eq),
        pl.Series("drawdown", drawdown_series(eq)),
        pl.Series("gross_equity", equity_curve(portfolio["gross_ret"].to_numpy())),
    )

    metrics = summarise(
        net_returns=portfolio["net_ret"].to_numpy(),
        gross_returns=portfolio["gross_ret"].to_numpy(),
        positions=portfolio["position"].to_numpy(),
        costs=portfolio["cost"].to_numpy(),
        periods_per_year=periods_per_year,
        rf=rf,
    )

    return BacktestResult(
        panel=panel,
        portfolio=portfolio,
        metrics=metrics,
        config={
            "signal": type(signal).__name__,
            "execution": execution,
            "lag_bars": lag,
            "extra_lag": extra_lag,
            "costs": costs.describe(),
            "cost_model": costs,
            "tickers": tickers,
            "periods_per_year": periods_per_year,
            "rf": rf,
            "audited": bool(audit) and hasattr(signal, "predict_all"),
        },
    )
