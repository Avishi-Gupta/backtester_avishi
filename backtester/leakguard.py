"""Look-ahead detection.

The premise of this project is that a backtest is a measuring instrument, and
an instrument you have not tried to break is an instrument you do not trust.
So the harness attacks its own signals.

**The method: future perturbation.** Take the panel. Choose a cut point ``k``.
Replace every bar at or after ``k`` with different numbers -- scaled prices,
scrambled order, whatever, as long as rows ``0..k-1`` are untouched. Ask the
signal for its positions again. Compare the first ``k`` positions.

If the signal is causal, those ``k`` positions cannot have moved: they were
computed from data that did not change. If even one of them moved, the signal
read something that had not happened yet, and the audit names the row.

This is strictly stronger than reading the code. It catches:

* ``shift(-1)`` typos and off-by-one window boundaries;
* whole-column statistics (``x.mean()``, ``x.std()``, ``MinMaxScaler().fit()``)
  fitted on the full sample -- the single most common leak in ML-flavoured
  strategies, and the one that is invisible in a diff;
* joins that pulled a restated or as-of-today field back onto old rows;
* a model fitted once on everything and then "evaluated" out of sample.

It is the same failure mode as a contaminated train/test split, which is why
the tests in ``tests/test_lookahead.py`` are the ones worth reading first.
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
            f"perturbed. Row {index} is BEFORE the cut, so nothing it is allowed "
            f"to see was altered. This signal reads the future."
        )


def perturb_future(
    frame: pl.DataFrame,
    cut: int,
    seed: int = 0,
    mode: str = "scale",
) -> pl.DataFrame:
    """Return a copy of ``frame`` with rows ``>= cut`` altered and rows ``< cut``
    byte-identical.

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

    # 'ret' and any precomputed feature columns are downstream of prices, so
    # drop them here: if the signal wants them it must recompute them, and if
    # it recomputes them from perturbed prices, that is the leak we are hunting.
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
    """Audit a signal. Returns the leaks found (or raises on the first).

    Runs every (cut, mode) combination, because different leaks show up under
    different attacks: a look-ahead ``shift(-1)`` fails ``shuffle`` immediately,
    while a full-sample normalisation only moves under ``scale``.
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
    """Cheap gate the engine runs before trusting a vectorised signal.

    One cut, two attack modes. Full paranoia belongs in the test suite; this is
    the seatbelt that fires on every run.
    """
    detect_lookahead(signal, frame, cuts=(0.6,), modes=("scale", "shuffle"), raise_on_leak=True)


def lag_sensitivity(
    bars: pl.DataFrame,
    signal: Any,
    lags: Sequence[int] = (0, 1, 2, 3, 5),
    **kwargs: Any,
) -> pl.DataFrame:
    """Sharpe as a function of extra execution latency.

    The diagnostic that separates a real, slow-decaying effect from a fitted
    one. A signal whose Sharpe collapses when you delay it by a single bar was
    almost certainly reading something close to the fill price.
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
