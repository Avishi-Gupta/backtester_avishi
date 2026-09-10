"""The yfinance loader, tested without a network.

yfinance returns two different shapes depending on how many tickers you ask
for: a (field, ticker) column MultiIndex for several, and flat columns for one.
The single-ticker case is the one that breaks in production, because everybody
develops against a list and then runs a one-name sanity check.

Both shapes are exercised here against a stub, so `fetch_yfinance` is covered
offline and the test suite stays deterministic.
"""

import sys
import types

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtester import data


def _multi_frame(tickers, n=5):
    """What yfinance returns for several tickers: (field, ticker) MultiIndex."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", name="Date")
    cols = {}
    for i, t in enumerate(tickers):
        base = 100.0 + 10 * i + np.arange(n)
        cols[("Open", t)] = base
        cols[("High", t)] = base + 1.0
        cols[("Low", t)] = base - 1.0
        cols[("Close", t)] = base + 0.5
        cols[("Adj Close", t)] = base + 0.4
        cols[("Volume", t)] = np.full(n, 1e6)
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _flat_frame(n=4):
    """What yfinance returns for a single ticker: flat columns, no MultiIndex."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", name="Date")
    base = 100.0 + np.arange(n)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Adj Close": base + 0.4,
            "Volume": np.full(n, 1e6),
        },
        index=idx,
    )


@pytest.fixture
def stub_yfinance(monkeypatch):
    """Install a fake yfinance module; the loader imports it lazily."""

    def install(frame_fn):
        stub = types.SimpleNamespace(download=lambda tickers, **kw: frame_fn(tickers))
        monkeypatch.setitem(sys.modules, "yfinance", stub)

    return install


def test_multi_ticker_download_becomes_a_long_panel(stub_yfinance):
    stub_yfinance(lambda tks: _multi_frame(tks))
    df = data.fetch_yfinance(["AAPL", "MSFT"], start="2024-01-01")

    assert df.columns == list(data.REQUIRED_COLUMNS)
    assert df.height == 10
    assert sorted(df["ticker"].unique().to_list()) == ["AAPL", "MSFT"]
    assert df["date"].dtype == pl.Date
    # Sorted by (ticker, date), which the engine's .over("ticker") relies on.
    assert df["ticker"].to_list() == ["AAPL"] * 5 + ["MSFT"] * 5
    assert df.filter(pl.col("ticker") == "MSFT")["date"].is_sorted()
    row = df.filter((pl.col("ticker") == "MSFT")).head(1)
    assert row["open"][0] == pytest.approx(110.0)
    assert row["adj_close"][0] == pytest.approx(110.4)


def test_single_ticker_flat_columns_are_handled(stub_yfinance):
    """The shape that silently differs. Without the MultiIndex fix-up this
    raises a KeyError deep inside the rename."""
    stub_yfinance(lambda tks: _flat_frame())
    df = data.fetch_yfinance(["AAPL"], start="2024-01-01")

    assert df.height == 4
    assert df["ticker"].unique().to_list() == ["AAPL"]
    assert df["close"][0] == pytest.approx(100.5)


def test_downloaded_bars_are_validated(stub_yfinance):
    """A bad download must fail at the door, not 200 lines later in the engine."""

    def broken(tks):
        f = _multi_frame(tks)
        f[("High", tks[0])] = f[("Low", tks[0])] - 5.0  # high < low
        return f

    stub_yfinance(broken)
    with pytest.raises(data.DataError, match="high < low"):
        data.fetch_yfinance(["AAPL"], start="2024-01-01")


def test_empty_download_raises_rather_than_returning_nothing(stub_yfinance):
    stub_yfinance(lambda tks: pd.DataFrame())
    with pytest.raises(data.DataError, match="no rows"):
        data.fetch_yfinance(["NOPE"], start="2024-01-01")


def test_cache_roundtrips_through_parquet(stub_yfinance, tmp_path):
    """The cache is what every downstream script reads, so prove the roundtrip
    preserves the schema exactly."""
    stub_yfinance(lambda tks: _multi_frame(tks))
    cache = tmp_path / "bars.parquet"
    written = data.fetch_yfinance(["AAPL", "MSFT"], start="2024-01-01", cache=cache)

    assert cache.exists()
    reloaded = data.scan_parquet(cache).collect()
    assert reloaded.schema == written.schema
    assert reloaded.equals(written)

    # And the full front door works on it, alignment included.
    panel = data.load(cache)
    assert panel.height == written.height
