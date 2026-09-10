"""positions -> fills -> PnL. This is where the one rule lives.

    A decision made using information from bar t can only be executed at t+1.

That rule is enforced *here*, not in user code, because a rule enforced in user
code is a rule that gets forgotten in the third notebook. The engine shifts the
signal's output forward before it ever touches a return, so a signal author
cannot opt out of the latency, cannot accidentally omit it, and cannot tune it
to make a result look better.

Timeline for the default ``execution="next_open"``:

    close of bar t-1   signal sees rows 0..t-1 and returns a target position
    open  of bar t     the engine fills that target
    open  of bar t+1   the position is marked out; ret_t = open_{t+1}/open_t - 1

So ``held_t = raw_{t-1}`` -- exactly one bar of latency, and the fill price is
never a price the signal looked at.

For ``execution="close_to_close"`` the return earned during bar t is
``close_t/close_{t-1} - 1``, which requires the fill to have happened at
``close_{t-1}``. A decision taken at that same close and filled at that same
close is the classic fantasy fill, so this mode carries TWO bars of latency
(``held_t = raw_{t-2}``): decide at close t-2, fill at close t-1. It is more
conservative than the industry default, on purpose.
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

#: Bars of latency each execution rule imposes. Not user-configurable; the
#: ``extra_lag`` argument can only ADD to it.
BASE_LAG: dict[str, int] = {"next_open": 1, "close_to_close": 2}


@dataclass
class BacktestResult:
    """Everything a run produced, plus enough provenance to reproduce it."""

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
    """Call ``predict`` once per bar, handing over only rows 0..t.

    Slicing a Polars DataFrame is a zero-copy view, so this is O(n) frames, not
    O(n^2) bytes. It is still one Python call per bar, which is the price of an
    interface where look-ahead cannot be written down.
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
        Additional bars of delay on top of the mandatory execution lag. Use it
        to answer "how fast does this decay?" -- a signal whose Sharpe halves
        with one extra day of latency is not a signal you can trade.
    audit:
        Whether to run the future-perturbation audit on a vectorised signal.
        Turning it off is allowed and is recorded in ``result.config`` so the
        report can say so out loud.
    """
    if extra_lag < 0:
        raise ValueError("extra_lag must be >= 0; negative latency is time travel")
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
        # ---- THE RULE. One line, and it is the whole point of the module. ----
        pl.col("raw_position").shift(lag).over("ticker").fill_null(0.0).alias("position")
    ).with_columns(
        # An untradable bar (no price, no return) carries the position and
        # accrues nothing -- it does not silently become a flat, costless day.
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

    # Equal-weight across names, so the portfolio's gross exposure stays <= 1
    # regardless of how many tickers are in the panel. Any other weighting is a
    # portfolio-construction decision and does not belong in the engine.
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
