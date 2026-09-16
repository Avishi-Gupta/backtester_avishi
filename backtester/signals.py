"""Example signals.

Four work correctly. Three are deliberately broken and are used as fixtures for
the leak-detection tests.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from backtester import features as F
from backtester.signal import BaseSignal

# ---------------------------------------------------------------------------
# Working signals
# ---------------------------------------------------------------------------


class BuyAndHold(BaseSignal):
    """Always fully long. Benchmark for the other signals."""

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        return pl.Series("position", np.ones(frame.height))


class ZeroSignal(BaseSignal):
    """Always flat. Produces zero turnover and zero cost."""

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        return pl.Series("position", np.zeros(frame.height))


class Momentum(BaseSignal):
    """Long if the trailing ``lookback``-bar return is positive, short if not.

    ``smooth`` replaces the hard sign with a tanh of the move's trailing
    Sharpe, which lowers turnover at a small cost in Sharpe.
    """

    def __init__(self, lookback: int = 60, smooth: bool = True, scale: float = 3.0):
        self.lookback = lookback
        self.smooth = smooth
        self.scale = scale
        self.warmup = lookback + 1

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        close = frame["close"].to_numpy()
        n = close.size
        pos = np.zeros(n)
        k = self.lookback
        if n <= k:
            return pl.Series("position", pos)
        mom = np.full(n, np.nan)
        mom[k:] = close[k:] / close[:-k] - 1.0
        if self.smooth:
            vol = _rolling_std(np.diff(np.log(close), prepend=np.log(close[0])), k)
            denom = np.where(vol > 0, vol * np.sqrt(k), np.nan)
            pos = np.tanh(self.scale * np.nan_to_num(mom / denom))
        else:
            pos = np.sign(np.nan_to_num(mom))
        pos[:k] = 0.0
        return pl.Series("position", pos)


class MeanReversion(BaseSignal):
    """Fade deviations from a trailing mean, sized by trailing z-score."""

    def __init__(self, window: int = 20, scale: float = 0.5):
        self.window = window
        self.scale = scale
        self.warmup = window

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        close = frame["close"].to_numpy()
        w = self.window
        mu = _rolling_mean(close, w)
        sd = _rolling_std(close, w)
        z = np.where(sd > 0, (close - mu) / sd, 0.0)
        pos = np.clip(-self.scale * np.nan_to_num(z), -1.0, 1.0)
        pos[:w] = 0.0
        return pl.Series("position", pos)


class VolFilteredMomentum(Momentum):
    """Momentum, switched off when trailing volatility is in its own top decile.

    The threshold is an expanding quantile, not a full-sample one. That is the
    only difference between this class and :class:`PeekingVolFilter` below.
    """

    def __init__(self, lookback: int = 60, vol_window: int = 20, quantile: float = 0.9, **kw):
        super().__init__(lookback=lookback, **kw)
        self.vol_window = vol_window
        self.quantile = quantile

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        base = super().predict_all(frame).to_numpy()
        close = frame["close"].to_numpy()
        lr = np.diff(np.log(close), prepend=np.log(close[0]))
        vol = _rolling_std(lr, self.vol_window)
        thresh = _expanding_quantile(vol, self.quantile, min_periods=250)
        gate = np.where(np.isnan(thresh), 1.0, (vol <= thresh).astype(float))
        return pl.Series("position", base * gate)


# ---------------------------------------------------------------------------
# Deliberately broken signals: fixtures for the leak-detection tests
# ---------------------------------------------------------------------------


class Oracle(BaseSignal):
    """Positioned on the sign of the return it is about to earn.

    Under the engine's default rule, ``held_t = raw_{t-1}`` and the return
    earned during bar ``t`` is ``open_{t+1}/open_t - 1``. To be perfectly
    positioned we therefore need ``raw_t = sign(open_{t+2} - open_{t+1})``.

    Used by the perfect-foresight test: a correctly aligned engine gives this
    a very high Sharpe, and a lower figure indicates a bar misalignment. Run
    with ``audit=False``, since the audit is designed to reject it.
    """

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        o = frame["open"].to_numpy()
        n = o.size
        fut = np.full(n, np.nan)
        if n > 2:
            fut[: n - 2] = o[2:] - o[1:-1]
        return pl.Series("position", np.sign(np.nan_to_num(fut)))


class PeekAheadMomentum(Momentum):
    """Momentum computed with a window that includes tomorrow's close.

    An off-by-one error: ``shift(-1)`` intended to mean "next row's feature"
    returns the next row's price instead.
    """

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        close = frame["close"].to_numpy()
        shifted = np.concatenate([close[1:], close[-1:]])  # tomorrow's close
        peeked = frame.with_columns(pl.Series("close", shifted))
        return super().predict_all(peeked)


class PeekingVolFilter(VolFilteredMomentum):
    """Momentum gated by a FULL-SAMPLE volatility quantile.

    Nothing is shifted backwards here. The threshold is computed once over the
    entire history, so the definition of "high volatility" in 2007 already
    reflects 2008. Equivalent to fitting a scaler before a train/test split.
    """

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        base = Momentum.predict_all(self, frame).to_numpy()
        close = frame["close"].to_numpy()
        lr = np.diff(np.log(close), prepend=np.log(close[0]))
        vol = _rolling_std(lr, self.vol_window)
        thresh = np.nanquantile(vol, self.quantile)  # full-sample: the leak
        gate = (vol <= thresh).astype(float)
        return pl.Series("position", base * np.nan_to_num(gate))


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    s = pl.Series(x).rolling_mean(w)
    return s.to_numpy()


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    s = pl.Series(x).rolling_std(w)
    return np.nan_to_num(s.to_numpy(), nan=0.0)


def _expanding_quantile(x: np.ndarray, q: float, min_periods: int = 250) -> np.ndarray:
    """Quantile of the values seen up to each point in time.

    O(n log n) via an incrementally sorted list; the naive version is O(n^2).
    """
    import bisect

    out = np.full(x.size, np.nan)
    seen: list[float] = []
    for i, v in enumerate(x):
        if np.isfinite(v):
            bisect.insort(seen, float(v))
        if len(seen) >= min_periods:
            idx = min(int(q * (len(seen) - 1)), len(seen) - 1)
            out[i] = seen[idx]
    return out


ALL_SIGNALS = {
    "buy_and_hold": BuyAndHold,
    "momentum": Momentum,
    "mean_reversion": MeanReversion,
    "vol_filtered_momentum": VolFilteredMomentum,
}

BROKEN_SIGNALS = {
    "oracle": Oracle,
    "peek_ahead_momentum": PeekAheadMomentum,
    "peeking_vol_filter": PeekingVolFilter,
}
