"""The tests that justify the project.

Test 1 (perfect foresight) proves the engine's plumbing is correct: if a signal
that knows the answer cannot score well, the engine is broken and nothing else
it says can be believed.

Test 2 (look-ahead detection) proves the guard works: a signal that peeks must
be caught, and if it is somehow let through, its edge must evaporate the moment
the execution lag is applied honestly.

They are complementary. The first one says "the instrument responds to signal";
the second says "the instrument is not just measuring itself".
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
    """A signal that knows the sign of the return it will earn must produce an
    absurd Sharpe. If this fails, the position/return alignment is off by a bar
    and every honest backtest is silently wrong too."""
    res = run_backtest(single, signals.Oracle(), costs=FRICTIONLESS, audit=False)
    honest = run_backtest(single, signals.Momentum(), costs=FRICTIONLESS)

    assert res.metrics.sharpe > 10, (
        f"oracle only scored {res.metrics.sharpe:.2f}; the engine is misaligning "
        "positions and returns"
    )
    assert res.metrics.sharpe > 10 * abs(honest.metrics.sharpe), (
        "perfect foresight must dominate an honest signal by an order of magnitude"
    )
    assert res.metrics.hit_rate > 0.95
    assert res.metrics.max_drawdown > -0.05


def test_perfect_foresight_is_destroyed_by_realistic_costs(single):
    """Same oracle, real costs. It flips from a fantasy to a losing strategy,
    because it trades every single day. This is the cheapest possible
    demonstration that gross results are not results."""
    gross = run_backtest(single, signals.Oracle(), costs=FRICTIONLESS, audit=False)
    net = run_backtest(
        single, signals.Oracle(), costs=CostModel(half_spread_bps=25, commission_bps=5), audit=False
    )
    assert net.metrics.sharpe < gross.metrics.sharpe
    assert net.metrics.turnover > 1.0  # flips position most days


# --------------------------------------------------------------------------
# 2. Look-ahead detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    [signals.Oracle, signals.PeekAheadMomentum, signals.PeekingVolFilter],
    ids=["oracle", "off_by_one_shift", "full_sample_quantile"],
)
def test_leaking_signals_are_caught(single, broken):
    """Every deliberately broken signal must be caught by the audit."""
    with pytest.raises(LeakageError):
        detect_lookahead(broken(), single)


def test_engine_refuses_to_run_a_leaking_signal(single):
    """The guard is on by default: you have to explicitly ask for the unaudited
    path, and that choice is recorded in the result's config."""
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
    """The guard must not cry wolf. A false positive here would make the whole
    mechanism something people switch off."""
    assert detect_lookahead(honest(), single, raise_on_leak=False) == []


def test_peeking_edge_collapses_once_lagged_honestly(single):
    """Even if a leak slipped past review, the lag-sensitivity diagnostic
    exposes it: an edge that is really tomorrow's price dies with one extra bar
    of latency, while a real effect decays gently."""
    fake = lag_sensitivity(single, signals.PeekAheadMomentum(lookback=5), lags=(0, 1), audit=False)
    real = lag_sensitivity(single, signals.Momentum(lookback=60), lags=(0, 1), audit=False)

    fake_drop = fake["sharpe_net"][0] - fake["sharpe_net"][1]
    real_drop = abs(real["sharpe_net"][0] - real["sharpe_net"][1])

    assert fake["sharpe_net"][0] > 3.0, "the peeking signal should look great at lag 0"
    assert fake_drop > real_drop, (
        "a leaked edge must decay faster with latency than a genuine one"
    )


def test_row_wise_interface_cannot_see_the_future(single):
    """Structural, not statistical: the safe interface hands over a slice, so
    the future is not in the object the signal receives."""

    seen: list[int] = []

    class Spy:
        def fit(self, train):
            pass

        def predict(self, history):
            seen.append(history.height)
            return 0.0

    sub = single.head(50)
    run_backtest(sub, Spy())
    # Row t was given exactly t+1 rows, never more.
    assert seen == list(range(1, 51))


def test_negative_lag_is_rejected(single):
    """You cannot ask the engine for negative latency."""
    with pytest.raises(ValueError):
        run_backtest(single, signals.Momentum(), extra_lag=-1)


def test_positions_are_shifted_by_exactly_one_bar(single):
    """The rule itself, asserted directly against the panel the engine emits."""
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
