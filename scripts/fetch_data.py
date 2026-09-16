#!/usr/bin/env python3
"""Download bars once and cache them to parquet.

    python scripts/fetch_data.py --tickers AAPL MSFT XOM JPM BAC SPY --start 2005-01-01

Everything downstream reads ``data_cache/bars.parquet``, so backtests run
offline and reproducibly. Re-running changes results, since adjusted history is
restated; see the point-in-time policy in backtester/data.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `python scripts/foo.py` work from any working directory without needing
# an editable install: put the repo root on sys.path before importing the
# package. `pip install -e .` also works and is preferred for real use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



def _require(module: str, extra: str = "") -> None:
    """Exit with the install command instead of a bare ModuleNotFoundError."""
    import importlib

    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parent.parent
        sys.exit(
            f"\nMissing dependency: {module}\n\n"
            f"Install with the same interpreter that runs this script:\n\n"
            f"    cd {root}\n"
            f"    python3 -m pip install -e .{extra}\n\n"
            f"`python3 -m pip` is used rather than a bare `pip` because the two can\n"
            f"resolve to different environments.\n"
        )


import argparse

_require("polars")
_require("yfinance", "[data]")

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
