"""Metrics against closed-form answers, and walk-forward hygiene."""

import numpy as np
import polars as pl
import pytest

from backtester import features as F
from backtester import signals
from backtester.data import synthetic_panel
from backtester.metrics import (
    equity_curve,
    exposure,
    hit_rate,
    max_drawdown,
    sharpe,
    turnover,
)
from backtester.walkforward import walk_forward


def test_sharpe_of_a_constant_series_is_zero_not_infinite():
    assert sharpe(np.full(100, 0.001)) == 0.0


def test_sharpe_matches_the_formula():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 5000)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)


def test_max_drawdown_known_answer():
    eq = np.array([1.0, 1.5, 0.75, 1.2])
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_max_drawdown_of_monotonic_curve_is_zero():
    assert max_drawdown(np.array([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_turnover_and_exposure_known_answer():
    pos = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    # |Δ| = [0, 1, 0, 2, 1] -> mean 0.8
    assert turnover(pos) == pytest.approx(0.8)
    assert exposure(pos) == pytest.approx(3 / 5)


def test_hit_rate_known_answer():
    assert hit_rate(np.array([1.0, -1.0, 1.0, 0.0])) == pytest.approx(0.5)


def test_equity_curve_compounds():
    np.testing.assert_allclose(equity_curve(np.array([0.1, -0.1])), [1.1, 0.99])


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


def test_walk_forward_returns_only_out_of_sample_bars(bars):
    wf = walk_forward(bars, lambda: signals.Momentum(lookback=20), train_bars=400, test_bars=100)
    n_dates = bars["date"].n_unique()
    expected = ((n_dates - 400) // 100) * 100
    assert wf.portfolio.height == expected
    assert wf.portfolio["date"].min() >= bars["date"].unique().sort()[400]
    assert wf.portfolio["date"].is_sorted()


def test_walk_forward_segments_do_not_overlap(bars):
    wf = walk_forward(bars, lambda: signals.Momentum(lookback=20), train_bars=400, test_bars=100)
    assert wf.portfolio["date"].n_unique() == wf.portfolio.height
    for a, b in zip(wf.folds, wf.folds[1:]):
        assert a.test_end == b.test_start


def test_walk_forward_refits_a_fresh_signal_each_fold(bars):
    """A signal instance reused across folds would carry fold-1 state into
    fold 2. The factory contract is what prevents it, so assert the factory is
    actually called once per fold."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return signals.Momentum(lookback=20)

    wf = walk_forward(bars, factory, train_bars=400, test_bars=200)
    assert calls["n"] == len(wf.folds)


def test_walk_forward_rejects_a_panel_that_is_too_short():
    tiny = synthetic_panel(n_days=50, tickers=("A",))
    with pytest.raises(ValueError, match="at least"):
        walk_forward(tiny, lambda: signals.Momentum(), train_bars=750, test_bars=125)


# --------------------------------------------------------------------------
# Feature causality
# --------------------------------------------------------------------------


def test_standard_features_are_causal(single):
    F.assert_causal(lambda df: F.standard_features(df), single)


def test_assert_causal_catches_a_leaking_feature(single):
    def leaky(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("close").shift(-1).alias("tomorrow"))

    with pytest.raises(AssertionError, match="reads ahead"):
        F.assert_causal(leaky, single)


def test_negative_lag_helper_raises():
    with pytest.raises(ValueError, match="future"):
        F.lag(pl.col("close"), -1)
