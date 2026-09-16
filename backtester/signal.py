"""Signal interfaces.

Two interfaces are provided.

:class:`Signal` is row-wise. The engine passes a frame containing rows ``0..t``
only, so later bars are not available to the signal. The cost is one Python
call per bar.

:class:`VectorSignal` receives the whole panel and returns all positions at
once, which runs roughly 100-1000x faster. Because that interface can read
future rows, the engine runs the future-perturbation audit in
:mod:`backtester.leakguard` before using its output.

Both return a target position in ``[-1, 1]``, not a trade. Sizing, netting,
execution lag and costs are handled by the engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


class PositionError(ValueError):
    """Raised when a signal returns something that is not a legal position."""


@runtime_checkable
class Signal(Protocol):
    """Row-wise signal. Receives rows 0..t only."""

    def fit(self, train: pl.DataFrame) -> None:
        """Optional calibration on a training slice.

        Called by the walk-forward runner with the in-sample window only.
        Stateless signals can implement this as ``pass``.
        """
        ...

    def predict(self, history: pl.DataFrame) -> float:
        """Target position in [-1, 1] for the bar following the last row.

        ``history`` contains rows up to and including the decision bar. The
        engine applies the execution lag afterwards, so this position is filled
        at the next bar's open.
        """
        ...


@runtime_checkable
class VectorSignal(Protocol):
    """Vectorised signal. Audited by the engine before its output is used."""

    def fit(self, train: pl.DataFrame) -> None: ...

    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        """Target positions for every row of ``frame``, same length, same order.

        Row ``t`` of the output must be computable from rows ``0..t`` of the
        input. The engine verifies this before using the result.
        """
        ...


class BaseSignal:
    """Base class providing a no-op ``fit`` and position clipping."""

    #: Bars of history required before the signal produces a non-zero position.
    warmup: int = 0

    def fit(self, train: pl.DataFrame) -> None:  # noqa: D102
        return None

    @staticmethod
    def clip(x: float) -> float:
        """Clamp to [-1, 1]. NaN becomes 0.0 rather than propagating into the
        equity curve."""
        import math

        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(-1.0, min(1.0, float(x)))


def validate_positions(pos: pl.Series) -> pl.Series:
    """Validate target positions at the engine boundary.

    Nulls become 0.0, so a signal still in its warmup period is flat rather
    than undefined. Infinite or out-of-range values raise.
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
