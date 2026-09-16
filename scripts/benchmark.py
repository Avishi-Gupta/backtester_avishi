#!/usr/bin/env python3
"""Polars vs pandas on the same feature workload.

Three implementations of an identical feature set (momentum, SMA, z-score and
realised volatility at four windows, grouped by ticker):

  pandas         groupby().rolling(), the idiomatic version
  polars_eager   expression API, executed immediately
  polars_lazy    same expressions, one optimised query plan, collected once
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
import time

_require("polars")
_require("pandas")

import numpy as np
import polars as pl

from backtester import data
from backtester import features as F

WINDOWS = (5, 20, 60, 120)


def pandas_features(pdf):
    import pandas as pd

    out = pdf.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", group_keys=False)["close"]
    out["log_ret"] = np.log(out["close"] / g.shift(1))
    for w in WINDOWS:
        out[f"mom_{w}"] = out["close"] / g.shift(w) - 1.0
        out[f"sma_{w}"] = g.transform(lambda s, w=w: s.rolling(w).mean())
        sd = g.transform(lambda s, w=w: s.rolling(w).std())
        out[f"z_{w}"] = (out["close"] - out[f"sma_{w}"]) / sd
        lr = out.groupby("ticker", group_keys=False)["log_ret"]
        out[f"vol_{w}"] = lr.transform(lambda s, w=w: s.rolling(w).std()) * np.sqrt(252)
    return out


def polars_eager_features(df: pl.DataFrame) -> pl.DataFrame:
    return F.standard_features(df, windows=WINDOWS, collect=True)


def polars_lazy_features(lf: pl.LazyFrame) -> pl.DataFrame:
    return F.standard_features(lf, windows=WINDOWS, collect=False).collect()


def timeit(fn, *args, repeat: int = 5) -> tuple[float, float]:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", default=None, help="parquet cache; synthetic if omitted")
    ap.add_argument("--n-days", type=int, default=5000)
    ap.add_argument("--tickers", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if args.bars:
        df = data.scan_parquet(args.bars).collect()
    else:
        names = [f"T{i:02d}" for i in range(args.tickers)]
        df = data.synthetic_panel(tickers=names, n_days=args.n_days)

    pdf = df.to_pandas()
    tmp = Path("/tmp/_bench_bars.parquet")
    df.write_parquet(tmp)

    print(f"panel: {df.height:,} rows x {df.width} cols "
          f"({df['ticker'].n_unique()} tickers x {df['date'].n_unique():,} days)")
    print(f"repeats: {args.repeat} (median reported)\n")

    results = {}
    results["pandas"] = timeit(pandas_features, pdf, repeat=args.repeat)
    results["polars_eager"] = timeit(polars_eager_features, df, repeat=args.repeat)
    results["polars_lazy"] = timeit(polars_lazy_features, df.lazy(), repeat=args.repeat)
    results["polars_scan_parquet"] = timeit(
        lambda: polars_lazy_features(pl.scan_parquet(tmp)), repeat=args.repeat
    )

    base = results["pandas"][0]
    print(f"{'implementation':<24} {'median (s)':>12} {'sd':>8} {'speedup':>9}")
    print("-" * 56)
    for name, (med, sd) in results.items():
        print(f"{name:<24} {med:>12.4f} {sd:>8.4f} {base / med:>8.1f}x")

    # Check the implementations agree before comparing their timings.
    a = polars_eager_features(df).sort(["ticker", "date"])["z_20"].to_numpy()
    b = pandas_features(pdf).sort_values(["ticker", "date"])["z_20"].to_numpy()
    ok = np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9)
    print(f"\nz_20 agrees between implementations: {ok}")


if __name__ == "__main__":
    main()
