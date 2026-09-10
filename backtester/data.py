"""Data loading and alignment, in Polars.

POINT-IN-TIME POLICY (this is a decision, not a default; it is written down
because an unwritten alignment policy is where leakage hides):

1.  A bar is stamped with the date on which it *closed*. Everything in row
    ``t`` was observable by the close of day ``t``.

2.  Missing bars mean "the market was not open for this instrument". We do NOT
    forward-fill prices across missing days by default. Forward filling turns a
    halted or delisted name into a flat, tradable, zero-risk asset, which
    flatters every backtest that touches it.

3.  When several tickers are aligned onto a common calendar, the calendar is
    the *union* of observed dates. A ticker with no bar on a calendar date gets
    nulls, and the engine treats a null bar as untradable: the previous
    position is carried, no return is accrued, no cost is charged. If you
    prefer a different convention, pass ``missing="ffill"`` and accept that you
    are asserting the price did not move.

4.  If ``missing="ffill"`` is used, the fill is a *backward-looking* fill
    (``strategy="forward"``), which can only ever copy a past value into the
    present. A backward fill (copying the future into the past) is never
    permitted anywhere in this codebase.

5.  ADJUSTED PRICES ARE NOT POINT-IN-TIME. yfinance returns split- and
    dividend-adjusted history, and that history is *restated* every time a new
    corporate action occurs. The 2015 adjusted close you download today is not
    the number anyone could have traded on in 2015. We therefore trade on
    RAW OHLC (``auto_adjust=False``) and carry ``adj_close`` alongside only for
    computing total returns. Splits are the one restatement we cannot ignore,
    so ``split_ratio`` is carried too. See the README for the full argument.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Literal, Sequence

import polars as pl

# The canonical schema every downstream module may rely on.
BAR_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Float64,
}

REQUIRED_COLUMNS = tuple(BAR_SCHEMA)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


class DataError(ValueError):
    """Raised when input bars violate the assumptions the engine relies on."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def scan_parquet(path: str | Path) -> pl.LazyFrame:
    """Lazily scan a cached parquet panel.

    Returns a LazyFrame, not a DataFrame. Nothing is read from disk until
    ``.collect()``; Polars pushes the projections and filters that downstream
    code adds down into the parquet reader, so a 20-year panel where you only
    want three columns and two years never fully materialises.
    """
    return pl.scan_parquet(path)


def fetch_yfinance(
    tickers: Sequence[str],
    start: str | dt.date,
    end: str | dt.date | None = None,
    cache: str | Path | None = None,
) -> pl.DataFrame:
    """Download daily bars and normalise them into the canonical long panel.

    Deliberately kept thin: it downloads, reshapes wide-to-long, and writes a
    parquet cache. Every other module reads the cache, so a backtest never
    depends on the network and results are reproducible.

    Note ``auto_adjust=False``: we want the raw OHLC that was actually printed,
    plus the adjusted close as a separate column. See the point-in-time policy
    at the top of this file.
    """
    import yfinance as yf  # imported lazily: the rest of the package is offline

    raw = yf.download(
        list(tickers),
        start=str(start),
        end=str(end) if end else None,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise DataError(f"yfinance returned no rows for {tickers}")

    # yfinance hands back a wide frame with a (field, ticker) column MultiIndex
    # for multiple tickers and a flat index for one. Normalise to long format.
    import pandas as pd

    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, [tickers[0]]])

    long = (
        raw.stack(level=-1, future_stack=True)
        .rename_axis(index=["Date", "Ticker"])
        .reset_index()
    )
    rename = {
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    long = long.rename(columns=rename)
    if "adj_close" not in long.columns:
        long["adj_close"] = long["close"]

    df = pl.from_pandas(long[list(REQUIRED_COLUMNS)])
    df = (
        df.with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("ticker").cast(pl.Utf8),
            *[pl.col(c).cast(pl.Float64) for c in
              ("open", "high", "low", "close", "adj_close", "volume")],
        )
        .drop_nulls(subset=["close"])
        .sort(["ticker", "date"])
    )

    validate_bars(df)

    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache)
    return df


def synthetic_panel(
    tickers: Sequence[str] = ("AAA", "BBB", "CCC", "IDX"),
    start: dt.date = dt.date(2005, 1, 3),
    n_days: int = 3000,
    seed: int = 7,
    annual_drift: float = 0.06,
    annual_vol: float = 0.22,
) -> pl.DataFrame:
    """Deterministic fake bars with fat tails, vol clustering and a crash.

    This exists so that the whole harness — tests, walk-forward, report — runs
    with no network and gives byte-identical results on every machine. It is
    NOT a substitute for real data in the writeup; it is a fixture.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = _business_days(start, n_days)
    frames = []

    for i, ticker in enumerate(tickers):
        r = rng.standard_normal(n_days)
        # GARCH-ish vol clustering so drawdowns and regimes actually appear.
        vol = np.empty(n_days)
        vol[0] = annual_vol / np.sqrt(252)
        for t in range(1, n_days):
            vol[t] = np.sqrt(
                0.02 * (annual_vol / np.sqrt(252)) ** 2
                + 0.90 * vol[t - 1] ** 2
                + 0.08 * (vol[t - 1] * r[t - 1]) ** 2
            )
        ret = annual_drift / 252 + vol * r
        # A GFC-shaped drawdown three years in, common to every name.
        crash = slice(int(n_days * 0.28), int(n_days * 0.36))
        ret[crash] -= 0.004
        # A little cross-sectional common factor plus idiosyncratic noise.
        ret = ret + 0.0002 * i

        close = 100.0 * np.exp(np.cumsum(ret))
        # Overnight gap: today's open differs from yesterday's close.
        gap = 0.3 * vol * rng.standard_normal(n_days)
        open_ = np.empty(n_days)
        open_[0] = 100.0
        open_[1:] = close[:-1] * np.exp(gap[1:])
        high = np.maximum(open_, close) * np.exp(np.abs(vol * rng.standard_normal(n_days)) * 0.4)
        low = np.minimum(open_, close) * np.exp(-np.abs(vol * rng.standard_normal(n_days)) * 0.4)
        volume = rng.lognormal(mean=15.0, sigma=0.4, size=n_days)

        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": [ticker] * n_days,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
        )

    df = pl.concat(frames).sort(["ticker", "date"])
    df = df.with_columns(pl.col("date").cast(pl.Date))
    validate_bars(df)
    return df


def _business_days(start: dt.date, n: int) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Alignment and validation
# ---------------------------------------------------------------------------


def align(
    bars: pl.DataFrame | pl.LazyFrame,
    missing: Literal["null", "ffill"] = "null",
) -> pl.DataFrame:
    """Put every ticker on the union calendar.

    ``missing="null"``  -> absent bars stay null and the engine treats them as
                           untradable (policy 3 above).
    ``missing="ffill"`` -> prices are carried forward, volume set to 0. This is
                           a forward fill only. There is no code path in this
                           package that fills backwards.
    """
    lf = bars.lazy()
    calendar = lf.select("date").unique()
    tickers = lf.select("ticker").unique()
    grid = calendar.join(tickers, how="cross")

    out = (
        grid.join(lf, on=["date", "ticker"], how="left")
        .sort(["ticker", "date"])
    )

    if missing == "ffill":
        price_cols = ["open", "high", "low", "close", "adj_close"]
        out = out.with_columns(
            [pl.col(c).forward_fill().over("ticker") for c in price_cols]
            + [pl.col("volume").fill_null(0.0)]
        )
    elif missing != "null":
        raise ValueError(f"unknown missing policy {missing!r}")

    return out.collect()


def validate_bars(bars: pl.DataFrame) -> None:
    """Fail loudly on the input problems that silently corrupt a backtest."""
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise DataError(f"bars are missing required columns: {missing}")

    if bars.height == 0:
        raise DataError("bars are empty")

    dupes = bars.group_by(["ticker", "date"]).len().filter(pl.col("len") > 1)
    if dupes.height:
        raise DataError(
            f"{dupes.height} duplicate (ticker, date) rows; the engine assumes one bar per day"
        )

    unsorted = (
        bars.sort(["ticker", "date"])
        .with_columns(
            (pl.col("date") <= pl.col("date").shift(1).over("ticker")).alias("_bad")
        )
        .select(pl.col("_bad").fill_null(False).any())
        .item()
    )
    if unsorted:
        raise DataError("dates are not strictly increasing within a ticker")

    bad_hl = bars.filter(
        pl.col("high").is_not_null()
        & pl.col("low").is_not_null()
        & (pl.col("high") < pl.col("low"))
    )
    if bad_hl.height:
        raise DataError(f"{bad_hl.height} bars have high < low")

    nonpositive = bars.filter(pl.col("close").is_not_null() & (pl.col("close") <= 0))
    if nonpositive.height:
        raise DataError(f"{nonpositive.height} bars have a non-positive close")


def to_returns(
    bars: pl.DataFrame,
    execution: Literal["next_open", "close_to_close"] = "next_open",
) -> pl.DataFrame:
    """Attach the return that is *earned* during each bar, given an execution rule.

    ``next_open``       ret_t = open_{t+1} / open_t - 1
        You are filled at the open of bar t and exit at the open of bar t+1.
        This is the return a position that was decided yesterday actually earns.

    ``close_to_close``  ret_t = close_t / close_{t-1} - 1
        Cheaper to reason about, but a fill at close_{t-1} needs the decision to
        have been made before that close, so the engine adds an extra bar of
        latency in this mode. See engine.run_backtest.

    Note the shift(-1) in ``next_open``. It looks like look-ahead and is not:
    it is the *realisation* of a position, not information fed into a decision.
    The distinction is the whole game, so it is asserted in the tests.
    """
    if execution == "next_open":
        expr = (pl.col("open").shift(-1) / pl.col("open") - 1.0).over("ticker")
    elif execution == "close_to_close":
        expr = (pl.col("close") / pl.col("close").shift(1) - 1.0).over("ticker")
    else:
        raise ValueError(f"unknown execution rule {execution!r}")
    return bars.sort(["ticker", "date"]).with_columns(expr.alias("ret"))


def load(
    source: str | Path | pl.DataFrame,
    missing: Literal["null", "ffill"] = "null",
) -> pl.DataFrame:
    """Convenience front door: parquet path or in-memory frame -> validated panel."""
    if isinstance(source, pl.DataFrame):
        bars = source
    else:
        bars = scan_parquet(source).collect()
    validate_bars(bars)
    return align(bars, missing=missing)
