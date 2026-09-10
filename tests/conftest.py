import polars as pl
import pytest

from backtester import data


@pytest.fixture(scope="session")
def bars() -> pl.DataFrame:
    """Deterministic synthetic panel. Same on every machine, no network."""
    return data.synthetic_panel(n_days=1500, seed=7)


@pytest.fixture(scope="session")
def single(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.filter(pl.col("ticker") == "AAA")


@pytest.fixture
def five_bars() -> pl.DataFrame:
    """A hand-built five-bar panel with round numbers, for the known-answer test.

    open:  100, 110, 121, 121, 108.9
    Returns earned per bar under next_open (open_{t+1}/open_t - 1):
        bar0: +10%,  bar1: +10%,  bar2: 0%,  bar3: -10%,  bar4: undefined
    """
    return pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True
            ),
            "ticker": ["X"] * 5,
            "open": [100.0, 110.0, 121.0, 121.0, 108.9],
            "high": [101.0, 111.0, 122.0, 122.0, 110.0],
            "low": [99.0, 109.0, 120.0, 120.0, 108.0],
            "close": [100.0, 110.0, 121.0, 121.0, 108.9],
            "adj_close": [100.0, 110.0, 121.0, 121.0, 108.9],
            "volume": [1e6] * 5,
        }
    )
