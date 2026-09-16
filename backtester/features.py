"""Feature computation in Polars.

Functions here return Polars expressions rather than DataFrames, so they compose
into a single query and so causality is visible in the expression itself.

A feature is causal when its value at row ``t`` depends only on rows ``<= t``:
rolling windows, ``shift(k)`` for ``k > 0``, and cumulative aggregates. Whole-
column aggregates such as ``mean()`` or ``max()`` used as scalars are not
causal, since they read the entire history.

:func:`assert_causal` checks this empirically rather than by inspection.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

TRADING_DAYS = 252

# Expressions that can read later rows in a column.
FORBIDDEN_METHODS = (
    "shift(-",
    "backward_fill",
    "reverse",
    "cum_sum(reverse=True)",
)


# ---------------------------------------------------------------------------
# Primitive, causal building blocks
# ---------------------------------------------------------------------------


def log_return(col: str = "close", by: str = "ticker") -> pl.Expr:
    """One-bar log return."""
    return (pl.col(col) / pl.col(col).shift(1)).log().over(by).alias("log_ret")


def momentum(window: int, col: str = "close", by: str = "ticker") -> pl.Expr:
    """Total return over the trailing ``window`` bars, inclusive of the current bar."""
    return (pl.col(col) / pl.col(col).shift(window) - 1.0).over(by).alias(f"mom_{window}")


def rolling_mean(window: int, col: str = "close", by: str = "ticker") -> pl.Expr:
    return pl.col(col).rolling_mean(window).over(by).alias(f"sma_{window}")


def rolling_std(window: int, col: str = "close", by: str = "ticker") -> pl.Expr:
    return pl.col(col).rolling_std(window).over(by).alias(f"sd_{window}")


def zscore(window: int, col: str = "close", by: str = "ticker") -> pl.Expr:
    """Trailing standard deviations between the current value and its trailing
    mean."""
    mu = pl.col(col).rolling_mean(window)
    sd = pl.col(col).rolling_std(window)
    return ((pl.col(col) - mu) / sd).over(by).alias(f"z_{window}")


def realised_vol(
    window: int, col: str = "close", by: str = "ticker", annualise: bool = True
) -> pl.Expr:
    """Trailing realised volatility of log returns."""
    lr = (pl.col(col) / pl.col(col).shift(1)).log()
    vol = lr.rolling_std(window)
    if annualise:
        vol = vol * (TRADING_DAYS ** 0.5)
    return vol.over(by).alias(f"vol_{window}")


def lag(expr: pl.Expr, k: int = 1, by: str = "ticker") -> pl.Expr:
    """Explicit extra latency. Negative ``k`` raises."""
    if k < 0:
        raise ValueError("negative lag reads future rows; use k >= 0")
    return expr.shift(k).over(by)


# ---------------------------------------------------------------------------
# Convenience: build the standard feature frame
# ---------------------------------------------------------------------------


def standard_features(
    bars: pl.DataFrame | pl.LazyFrame,
    windows: tuple[int, ...] = (5, 20, 60, 120),
    collect: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """The feature set used by the example signals.

    Built lazily and collected once, so the repeated rolling computations over
    ``close`` are deduplicated and executed in a single pass.
    """
    lf = bars.lazy().sort(["ticker", "date"])
    exprs: list[pl.Expr] = [log_return()]
    for w in windows:
        exprs += [
            momentum(w),
            rolling_mean(w),
            zscore(w),
            realised_vol(w),
        ]
    out = lf.with_columns(exprs)
    return out.collect() if collect else out


# ---------------------------------------------------------------------------
# Causality check
# ---------------------------------------------------------------------------


def assert_causal(
    build: Callable[[pl.DataFrame], pl.DataFrame],
    bars: pl.DataFrame,
    n_perturb: int = 40,
    seed: int = 0,
    rtol: float = 1e-9,
) -> None:
    """Check a feature builder against future perturbation.

    Takes the panel, picks a cut point, perturbs every bar after the cut,
    rebuilds the features and compares the rows before the cut. Any change
    means a feature read data from after the cut.

    Detects whole-column aggregates, off-by-one errors in ``shift``, and joins
    made on the wrong side.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    cut = bars.height // 2

    baseline = build(bars)

    scrambled = bars.clone()
    tail = scrambled.slice(cut, scrambled.height - cut)
    noise = rng.uniform(1.5, 2.5, size=tail.height)
    tail = tail.with_columns(
        [(pl.col(c) * pl.Series(noise)) for c in ("open", "high", "low", "close", "adj_close")]
    )
    scrambled = pl.concat([scrambled.slice(0, cut), tail])

    perturbed = build(scrambled)

    a = baseline.slice(0, cut)
    b = perturbed.slice(0, cut)

    numeric = [c for c, d in zip(a.columns, a.dtypes) if d.is_numeric()]
    for col in numeric:
        x = a[col].to_numpy()
        y = b[col].to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(x) - np.nan_to_num(y)))
        scale = np.maximum(1.0, np.abs(np.nan_to_num(x)))
        if np.any(diff / scale > rtol):
            bad = int(np.argmax(diff / scale))
            raise AssertionError(
                f"feature {col!r} changed at row {bad} when only FUTURE bars were "
                f"perturbed ({x[bad]} -> {y[bad]}). This feature reads ahead."
            )
    _ = n_perturb  # kept for API compatibility with older call sites
