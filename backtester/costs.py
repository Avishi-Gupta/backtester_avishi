"""Transaction cost models.

A backtest without costs is a statement about the past that nobody could have
acted on. The default here is deliberately *not* zero, because a zero default
is how a gross-looking result reaches a slide deck.

All costs are expressed in **return units on the notional**, so they subtract
directly from the strategy's per-bar return. A position change of 1.0 (flat to
fully long) at 5bp all-in costs 0.0005.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Spread + commission + (optionally) participation-linear slippage.

    Parameters
    ----------
    half_spread_bps:
        You cross half the quoted spread on each side. For liquid US large caps
        1-2bp is defensible; for a small cap it is not.
    commission_bps:
        Broker fee per unit of notional traded.
    slippage_coef_bps:
        Market impact, charged as ``slippage_coef_bps * participation`` where
        participation is your traded notional divided by the bar's dollar
        volume. Linear impact is the crude model; square-root is the usual
        refinement and is available via ``impact_exponent``.
    impact_exponent:
        1.0 for linear impact, 0.5 for the square-root law.
    notional:
        Portfolio notional in currency units, used only to convert a position
        change into a dollar amount for the participation calculation.
    """

    half_spread_bps: float = 1.0
    commission_bps: float = 0.5
    slippage_coef_bps: float = 0.0
    impact_exponent: float = 1.0
    notional: float = 1_000_000.0

    @property
    def linear_bps(self) -> float:
        """The part of the cost that does not depend on size."""
        return self.half_spread_bps + self.commission_bps

    def per_bar_cost(self, traded: pl.Expr, dollar_volume: pl.Expr | None = None) -> pl.Expr:
        """Cost expression, given |Δposition| and (optionally) bar dollar volume.

        ``traded`` is the absolute change in target position for this bar,
        already in units of notional fraction.
        """
        cost = traded * (self.linear_bps * BPS)

        if self.slippage_coef_bps > 0.0:
            if dollar_volume is None:
                raise ValueError(
                    "slippage_coef_bps > 0 requires dollar volume; pass bars with "
                    "a 'volume' column, or set slippage_coef_bps=0 and say so in "
                    "the report."
                )
            participation = (traded * self.notional) / pl.max_horizontal(
                dollar_volume, pl.lit(1.0)
            )
            cost = cost + traded * (
                self.slippage_coef_bps * BPS * participation ** self.impact_exponent
            )
        return cost

    def scaled(self, factor: float) -> "CostModel":
        """Same model with every cost term multiplied -- used by the sensitivity table."""
        return CostModel(
            half_spread_bps=self.half_spread_bps * factor,
            commission_bps=self.commission_bps * factor,
            slippage_coef_bps=self.slippage_coef_bps * factor,
            impact_exponent=self.impact_exponent,
            notional=self.notional,
        )

    def describe(self) -> str:
        s = f"{self.linear_bps:.2f}bp linear"
        if self.slippage_coef_bps:
            s += f" + {self.slippage_coef_bps:.1f}bp x participation^{self.impact_exponent:g}"
        return s


#: The one you should have to argue your way out of, not into.
FRICTIONLESS = CostModel(half_spread_bps=0.0, commission_bps=0.0)
RETAIL = CostModel(half_spread_bps=2.5, commission_bps=1.0)
INSTITUTIONAL = CostModel(half_spread_bps=1.0, commission_bps=0.5, slippage_coef_bps=10.0)
