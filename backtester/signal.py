"""The interface a user implements.

There are two ways to write a signal, and the difference is the central design
decision of this harness.

**The safe interface** (:class:`Signal`) is row-wise. The engine hands you a
frame containing rows ``0..t`` and nothing else. You physically cannot read
tomorrow's close, because tomorrow's close is not in the object you were given.
Look-ahead is not detected here; it is *unrepresentable*. The cost is speed:
one Python call per bar.

**The fast interface** (:class:`VectorSignal`) hands you the whole panel and
asks for the whole position series at once, which is 100-1000x faster. That
convenience buys back the ability to cheat, so the engine refuses to run a
``VectorSignal`` until it has passed the future-perturbation audit in
:mod:`backtester.leakguard`. Speed is available; unaudited speed is not.

Both return a **target position** in ``[-1, 1]``, never a trade. Sizing,
netting, latency and costs belong to the engine, where the constraints live. A
signal that could emit "BUY 100 shares" would be a signal that has opinions
about execution, and those opinions would not be tested by anything.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


class PositionError(ValueError):
    """Raised when a signal returns something that is not a legal position."""


@runtime_checkable
class Signal(Protocol):
    """Row-wise signal. Cannot see the future by construction."""

    def fit(self, train: pl.DataFrame) -> None:
        """Optional calibration on a training slice.

        Called by the walk-forward runner with the in-sample window only. If
        your signal is stateless, implement this as ``pass``.
        """
        ...

    def predict(self, history: pl.DataFrame) -> float:
        """Target position in [-1, 1] for the bar that *follows* the last row.

        ``history`` contains rows up to and including the decision bar. The
        engine then applies its own execution lag on top, so the position you
        return here is filled at the next bar's open, not at the close you are
        looking at.
        """
        ...


@runtime_checkable
class VectorSignal(Protocol):
    """Vectorised signal. Fast, auditable, not trusted until audited."""

    def fit(self, train: pl.DataFrame) -> None: ...

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        """Target positions for every row of ``frame``, same length, same order.

        Row ``t`` of the output must be computable from rows ``0..t`` of the
        input. The engine verifies this claim rather than believing it.
        """
        ...


class BaseSignal:
    """Small convenience base. Stateless ``fit``, and position hygiene."""

    #: Bars of history required before the signal produces a non-zero position.
    warmup: int = 0

    def fit(self, train: pl.DataFrame) -> None:  # noqa: D102
        return None

    @staticmethod
    def clip(x: float) -> float:
        """Clamp to the legal range, and turn NaN into flat rather than into
        a silent NaN that poisons the whole equity curve downstream."""
        import math

        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(-1.0, min(1.0, float(x)))


def validate_positions(pos: pl.Series) -> pl.Series:
    """Reject illegal target positions loudly, at the boundary.

    Nulls become 0.0 (flat) because a signal with insufficient warmup should be
    out of the market, not undefined. Infinities and out-of-range values are
    programmer errors and raise.
    """
    filled = pos.fill_null(0.0).fill_nan(0.0)
    if filled.is_infinite().any():
        raise PositionError("signal returned an infinite position")
    lo, hi = filled.min(), filled.max()
    if lo is not None and (lo < -1.0 - 1e-12 or hi > 1.0 + 1e-12):
        raise PositionError(
            f"target positions must lie in [-1, 1]; got [{lo:.4f}, {hi:.4f}]. "
            "Leverage belongs in the engine's sizing, not in the signal."
        )
    return filled.cast(pl.Float64)
