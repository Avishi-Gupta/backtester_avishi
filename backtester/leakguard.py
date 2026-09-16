"""Look-ahead detection by future perturbation.

Method: take a panel, choose a cut point ``k``, replace every bar at or after
``k``, and recompute the signal's positions. Rows ``0..k-1`` were computed from
unchanged data, so a causal signal returns identical values for them. Any
difference means the signal read data from after the cut.

This detects:

* ``shift(-1)`` errors and off-by-one window boundaries;
* whole-column statistics (``x.mean()``, ``MinMaxScaler().fit()``) computed over
  the full sample;
* joins that bring a restated or as-of-today field onto historical rows;
* a model fitted on all data and then evaluated "out of sample".
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


class LeakageError(AssertionError):
    """Raised when a signal's past output depends on future data."""

    def __init__(self, index: int, before: float, after: float, cut: int, n: int):
        self.index = index
        self.cut = cut
        super().__init__(
            f"LOOK-AHEAD DETECTED: position at row {index} changed from "
            f"{before:.6g} to {after:.6g} when only rows >= {cut} (of {n}) were "
            f"perturbed. Row {index} precedes the cut, so none of the data it "
            f"may use was altered. This signal reads future rows."
        )


def perturb_future(
    frame: pl.DataFrame,
    cut: int,
    seed: int = 0,
    mode: str = "scale",
) -> pl.DataFrame:
    """Return a copy of ``frame`` with rows ``>= cut`` altered and rows
    ``< cut`` unchanged.

    ``mode="scale"``   multiply future prices by random factors in [1.5, 2.5]
    ``mode="shuffle"`` randomly permute the future rows' prices
    ``mode="flat"``    replace all future prices with the last pre-cut close
    """
    rng = np.random.default_rng(seed)
    head = frame.slice(0, cut)
    tail = frame.slice(cut, frame.height - cut)
    if tail.height == 0:
        return frame

    if mode == "scale":
        f = pl.Series(rng.uniform(1.5, 2.5, size=tail.height))
        tail = tail.with_columns([pl.col(c) * f for c in PRICE_COLUMNS if c in tail.columns])
    elif mode == "shuffle":
        idx = rng.permutation(tail.height)
        tail = tail.with_columns(
            [pl.col(c).gather(idx) for c in PRICE_COLUMNS if c in tail.columns]
        )
    elif mode == "flat":
        last = head["close"][-1] if head.height else float(tail["close"][0])
        tail = tail.with_columns(
            [pl.lit(float(last)).alias(c) for c in PRICE_COLUMNS if c in tail.columns]
        )
    else:
        raise ValueError(f"unknown perturbation mode {mode!r}")

    # 'ret' is derived from prices, so null it in the perturbed section: a
    # signal that needs it must recompute it from the perturbed prices.
    tail = tail.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in ("ret",) if c in tail.columns]
    )
    return pl.concat([head, tail])


def _positions(signal: Any, frame: pl.DataFrame) -> np.ndarray:
    if hasattr(signal, "predict_all"):
        return np.asarray(pl.Series(signal.predict_all(frame)).to_numpy(), dtype=float)
    n = frame.height
    return np.array([signal.predict(frame.slice(0, t + 1)) for t in range(n)], dtype=float)


def detect_lookahead(
    signal: Any,
    frame: pl.DataFrame,
    cuts: Sequence[float] = (0.4, 0.6, 0.8),
    modes: Sequence[str] = ("scale", "shuffle"),
    atol: float = 1e-10,
    seed: int = 0,
    raise_on_leak: bool = True,
) -> list[LeakageError]:
    """Audit a signal, returning the leaks found or raising on the first.

    Runs every (cut, mode) combination. Different leaks surface under different
    perturbations: a ``shift(-1)`` fails under ``shuffle``, while a full-sample
    normalisation only changes under ``scale``.
    """
    frame = frame.sort("date") if "date" in frame.columns else frame
    n = frame.height
    baseline = _positions(signal, frame)
    found: list[LeakageError] = []

    for i, frac in enumerate(cuts):
        cut = int(n * frac)
        if cut <= 1 or cut >= n:
            continue
        for j, mode in enumerate(modes):
            perturbed = perturb_future(frame, cut, seed=seed + 17 * i + j, mode=mode)
            after = _positions(signal, perturbed)

            a, b = baseline[:cut], after[:cut]
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
            bad = np.nonzero(diff > atol)[0]
            if bad.size:
                k = int(bad[0])
                err = LeakageError(k, float(a[k]), float(b[k]), cut, n)
                if raise_on_leak:
                    raise err
                found.append(err)
    return found


def audit_vector_signal(signal: Any, frame: pl.DataFrame) -> None:
    """Check run by the engine before using a vectorised signal's output.

    One cut point and two perturbation modes, to keep per-run cost low. The
    test suite runs the full grid.
    """
    detect_lookahead(signal, frame, cuts=(0.6,), modes=("scale", "shuffle"), raise_on_leak=True)


def lag_sensitivity(
    bars: pl.DataFrame,
    signal: Any,
    lags: Sequence[int] = (0, 1, 2, 3, 5),
    **kwargs: Any,
) -> pl.DataFrame:
    """Sharpe as a function of extra execution latency.

    A signal whose Sharpe collapses under one bar of added delay is likely
    reading prices close to its own fill price.
    """
    from backtester.engine import run_backtest

    rows = []
    for k in lags:
        res = run_backtest(bars, signal, extra_lag=k, **kwargs)
        rows.append(
            {
                "extra_lag_bars": k,
                "sharpe_net": res.metrics.sharpe,
                "sharpe_gross": res.metrics.gross_sharpe,
                "max_drawdown": res.metrics.max_drawdown,
            }
        )
    return pl.DataFrame(rows)
