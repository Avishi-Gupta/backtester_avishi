"""Execution-lag and leak-detection tests.

The perfect-foresight test checks that positions and returns are correctly
aligned: a signal that knows the sign of the return it earns must score very
highly, otherwise the engine is off by a bar.

The look-ahead tests check that leaking signals are caught, and that a leaked
edge does not survive the execution lag being applied correctly.
"""

import numpy as np
import polars as pl
import pytest

from backtester import signals
from backtester.costs import FRICTIONLESS, CostModel
from backtester.engine import run_backtest
from backtester.leakguard import LeakageError, detect_lookahead, lag_sensitivity


# --------------------------------------------------------------------------
# 1. Perfect foresight
# --------------------------------------------------------------------------


def test_perfect_foresight_scores_enormously(single):
    """A signal positioned on the sign of the return it earns should score very
    highly. A low figure indicates positions and returns are misaligned by a
    bar, which would affect every other result the engine produces."""
    res = run_backtest(single, signals.Oracle(), costs=FRICTIONLESS, audit=False)
    honest = run_backtest(single, signals.Momentum(), costs=FRICTIONLESS)

    assert res.metrics.sharpe > 10, (
        f"oracle scored {res.metrics.sharpe:.2f}; positions and returns are "
        "likely misaligned"
    )
    assert res.metrics.sharpe > 10 * abs(honest.metrics.sharpe), (
        "perfect foresight should exceed a working signal by an order of magnitude"
    )
    assert res.metrics.hit_rate > 0.95
    assert res.metrics.max_drawdown > -0.05


def test_perfect_foresight_is_destroyed_by_realistic_costs(single):
    """The same signal under realistic costs. It trades almost every day, so
    the cost charge removes the result."""
    gross = run_backtest(single, signals.Oracle(), costs=FRICTIONLESS, audit=False)
    net = run_backtest(
        single, signals.Oracle(), costs=CostModel(half_spread_bps=25, commission_bps=5), audit=False
    )
    assert net.metrics.sharpe < gross.metrics.sharpe
    assert net.metrics.turnover > 1.0  # changes position on most days


# --------------------------------------------------------------------------
# 2. Look-ahead detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    [signals.Oracle, signals.PeekAheadMomentum, signals.PeekingVolFilter],
    ids=["oracle", "off_by_one_shift", "full_sample_quantile"],
)
def test_leaking_signals_are_caught(single, broken):
    """Each deliberately broken signal should be caught by the audit."""
    with pytest.raises(LeakageError):
        detect_lookahead(broken(), single)


def test_engine_refuses_to_run_a_leaking_signal(single):
    """The audit runs by default. Skipping it requires audit=False, which is
    recorded in the result config."""
    with pytest.raises(LeakageError):
        run_backtest(single, signals.PeekAheadMomentum())

    ok = run_backtest(single, signals.PeekAheadMomentum(), audit=False)
    assert ok.config["audited"] is False


@pytest.mark.parametrize(
    "honest",
    [signals.Momentum, signals.MeanReversion, signals.VolFilteredMomentum, signals.BuyAndHold],
    ids=["momentum", "mean_reversion", "vol_filtered", "buy_and_hold"],
)
def test_honest_signals_pass_the_audit(single, honest):
    """Check for false positives: correct signals must pass the audit."""
    assert detect_lookahead(honest(), single, raise_on_leak=False) == []


def test_peeking_edge_collapses_once_lagged_honestly(single):
    """Lag sensitivity as a second check on leakage: a result driven by
    next-bar prices loses most of its Sharpe at one extra bar of latency,
    while a slower effect decays gradually."""
    fake = lag_sensitivity(single, signals.PeekAheadMomentum(lookback=5), lags=(0, 1), audit=False)
    real = lag_sensitivity(single, signals.Momentum(lookback=60), lags=(0, 1), audit=False)

    fake_drop = fake["sharpe_net"][0] - fake["sharpe_net"][1]
    real_drop = abs(real["sharpe_net"][0] - real["sharpe_net"][1])

    assert fake["sharpe_net"][0] > 3.0, "the peeking signal should score highly at lag 0"
    assert fake_drop > real_drop, (
        "a leaked result should decay faster with latency than a real one"
    )


def test_row_wise_interface_cannot_see_the_future(single):
    """The row-wise interface passes a slice, so later bars are not present in
    the object the signal receives."""

    seen: list[int] = []

    class Spy:
        def fit(self, train):
            pass

        def predict(self, history):
            seen.append(history.height)
            return 0.0

    sub = single.head(50)
    run_backtest(sub, Spy())
    # Row t received exactly t+1 rows.
    assert seen == list(range(1, 51))


def test_negative_lag_is_rejected(single):
    """Negative extra_lag is rejected."""
    with pytest.raises(ValueError):
        run_backtest(single, signals.Momentum(), extra_lag=-1)


def test_positions_are_shifted_by_exactly_one_bar(single):
    """Check the execution lag directly against the panel the engine emits."""
    res = run_backtest(single, signals.Momentum(), audit=False)
    raw = res.panel["raw_position"].to_numpy()
    held = res.panel["position"].to_numpy()
    assert held[0] == 0.0
    np.testing.assert_allclose(held[1:], raw[:-1], atol=1e-12)


def test_close_to_close_mode_carries_two_bars_of_latency(single):
    res = run_backtest(single, signals.Momentum(), execution="close_to_close", audit=False)
    assert res.config["lag_bars"] == 2
    raw = res.panel["raw_position"].to_numpy()
    held = res.panel["position"].to_numpy()
    np.testing.assert_allclose(held[2:], raw[:-2], atol=1e-12)
