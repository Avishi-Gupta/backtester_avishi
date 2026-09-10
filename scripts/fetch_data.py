#!/usr/bin/env python3
"""Download bars once, cache to parquet, never touch the network again.

    python scripts/fetch_data.py --tickers AAPL MSFT XOM JPM BAC SPY --start 2005-01-01

Everything downstream reads ``data_cache/bars.parquet``, so a backtest is
reproducible and offline. Re-run this when you want fresh data and accept that
your results will move: adjusted history is restated, which is itself one of the
point-in-time problems documented in backtester/data.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backtester import data

DEFAULT_TICKERS = ["AAPL", "MSFT", "XOM", "JPM", "BAC", "KO", "SPY"]
CACHE = Path(__file__).resolve().parent.parent / "data_cache" / "bars.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(CACHE))
    args = ap.parse_args()

    print(f"downloading {len(args.tickers)} tickers from {args.start} ...")
    df = data.fetch_yfinance(args.tickers, start=args.start, end=args.end, cache=args.out)
    print(f"  {df.height:,} bars, {df['ticker'].n_unique()} tickers, "
          f"{df['date'].min()} -> {df['date'].max()}")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
