"""Zero signal, cost monotonicity, and the hand-computed known answer."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from backtester import signals
from backtester.costs import FRICTIONLESS, CostModel
from backtester.data import DataError, align, synthetic_panel, to_returns, validate_bars
from backtester.engine import run_backtest


# --------------------------------------------------------------------------
# 3. Zero signal
# --------------------------------------------------------------------------


def test_zero_signal_has_zero_turnover_and_zero_cost(bars):
    res = run_backtest(bars, signals.ZeroSignal(), costs=CostModel(half_spread_bps=50))
    assert res.metrics.turnover == 0.0
    assert res.metrics.n_trades == 0
    assert res.portfolio["cost"].sum() == 0.0
    assert res.metrics.exposure == 0.0
    np.testing.assert_allclose(res.equity, 1.0)


def test_constant_nonzero_position_trades_once(bars):
    """Buy-and-hold pays the spread exactly once, on the way in."""
    res = run_backtest(bars, signals.BuyAndHold(), costs=CostModel(half_spread_bps=10))
    assert res.metrics.n_trades == 1
    total_cost = res.portfolio["cost"].sum()
    expected = (10.0 + 0.5) * 1e-4  # one full unit of position, all-in bps
    assert total_cost == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# 4. Cost monotonicity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("signal", [signals.Momentum, signals.MeanReversion])
def test_higher_costs_never_increase_net_pnl(bars, signal):
    prev = None
    for bps in (0.0, 1.0, 2.0, 5.0, 10.0, 25.0):
        res = run_backtest(
            bars, signal(), costs=CostModel(half_spread_bps=bps, commission_bps=0.0)
        )
        total = res.equity[-1]
        if prev is not None:
            assert total <= prev + 1e-12, f"net PnL rose when costs rose to {bps}bp"
        prev = total


def test_cost_scales_linearly_with_bps(bars):
    a = run_backtest(bars, signals.Momentum(), costs=CostModel(1.0, 0.0))
    b = run_backtest(bars, signals.Momentum(), costs=CostModel(3.0, 0.0))
    assert b.portfolio["cost"].sum() == pytest.approx(3 * a.portfolio["cost"].sum(), rel=1e-9)


# --------------------------------------------------------------------------
# 5. Known answer -- five bars, computed by hand
# --------------------------------------------------------------------------


def test_known_answer_five_bars(five_bars):
    """Fully long from the first tradable bar, no costs.

    opens: 100, 110, 121, 121, 108.9
    ret_t = open_{t+1}/open_t - 1  ->  [+0.10, +0.10, 0.00, -0.10, null->0]

    BuyAndHold emits raw_position = 1 on every bar. The engine shifts by one,
    so held = [0, 1, 1, 1, 1] and gross PnL per bar is
        [0*0.10, 1*0.10, 1*0.00, 1*(-0.10), 1*0] = [0, 0.10, 0, -0.10, 0]

    Equity = 1 * 1.10 * 1.00 * 0.90 * 1.00 = 0.99. Losing money on a round
    trip that ends where it started is correct: compounding is not additive.
    """
    res = run_backtest(five_bars, signals.BuyAndHold(), costs=FRICTIONLESS)

    np.testing.assert_allclose(
        res.panel["ret"].to_numpy(), [0.10, 0.10, 0.0, -0.10, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(res.panel["position"].to_numpy(), [0, 1, 1, 1, 1], atol=1e-12)
    np.testing.assert_allclose(
        res.portfolio["net_ret"].to_numpy(), [0.0, 0.10, 0.0, -0.10, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        res.portfolio["equity"].to_numpy(), [1.0, 1.10, 1.10, 0.99, 0.99], atol=1e-12
    )
    assert res.metrics.total_return == pytest.approx(-0.01, abs=1e-12)


def test_known_answer_with_costs(five_bars):
    """Same five bars, 10bp half-spread and 0 commission. One trade of size 1.0
    on bar 1, so exactly 0.001 comes out of that bar's return."""
    res = run_backtest(
        five_bars, signals.BuyAndHold(), costs=CostModel(half_spread_bps=10.0, commission_bps=0.0)
    )
    np.testing.assert_allclose(
        res.portfolio["cost"].to_numpy(), [0.0, 0.001, 0.0, 0.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        res.portfolio["net_ret"].to_numpy(), [0.0, 0.099, 0.0, -0.10, 0.0], atol=1e-12
    )


# --------------------------------------------------------------------------
# Data-layer guards
# --------------------------------------------------------------------------


def test_duplicate_bars_are_rejected(five_bars):
    dupe = pl.concat([five_bars, five_bars.head(1)])
    with pytest.raises(DataError, match="duplicate"):
        validate_bars(dupe)


def test_high_below_low_is_rejected(five_bars):
    bad = five_bars.with_columns(pl.col("high") - 100.0)
    with pytest.raises(DataError, match="high < low"):
        validate_bars(bad)


def test_alignment_does_not_fill_from_the_future():
    """A ticker missing its middle bar must not receive the NEXT bar's price."""
    panel = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)] * 2,
            "ticker": ["A"] * 3 + ["B"] * 3,
            "open": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "high": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "low": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "close": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "adj_close": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "volume": [1.0] * 6,
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    holed = panel.filter(~((pl.col("ticker") == "B") & (pl.col("date") == dt.date(2020, 1, 2))))
    filled = align(holed, missing="ffill").sort(["ticker", "date"])
    b = filled.filter(pl.col("ticker") == "B")["close"].to_list()
    assert b == [10.0, 10.0, 30.0], "forward fill must copy the PAST value, never the future"


def test_missing_bars_default_to_untradable():
    panel = synthetic_panel(n_days=60, tickers=("A", "B"))
    holed = panel.filter(
        ~((pl.col("ticker") == "B") & (pl.col("date") == panel["date"][10]))
    )
    aligned = align(holed, missing="null")
    row = aligned.filter((pl.col("ticker") == "B") & (pl.col("date") == panel["date"][10]))
    assert row["close"][0] is None

    res = run_backtest(aligned, signals.BuyAndHold())
    assert np.isfinite(res.equity).all()


def test_to_returns_next_open_matches_definition(five_bars):
    r = to_returns(five_bars, execution="next_open")["ret"].to_list()
    assert r[0] == pytest.approx(110 / 100 - 1)
    assert r[3] == pytest.approx(108.9 / 121 - 1)
    assert r[4] is None
