# backtester

A daily-bar backtesting harness for evaluating trading signals. It takes a
signal function and returns performance metrics net of transaction costs, with
execution lag and look-ahead checks enforced by the framework rather than left
to the signal author.

Pipeline:

```
data -> features -> signal -> target positions -> fills -> PnL -> metrics
```

## Install

```bash
git clone https://github.com/Avishi-Gupta/backtester_avishi.git
cd backtester_avishi
python3 -m pip install -e ".[data,dev]"
pytest -q
```

Requires Python 3.10+. The test suite is offline and takes about 2 seconds.

## Usage

```bash
# synthetic fixture data
python3 scripts/run_report.py

# real data: download once, then report
python3 scripts/fetch_data.py --tickers AAPL MSFT XOM JPM BAC KO SPY --start 2005-01-01
python3 scripts/run_report.py --bars data_cache/bars.parquet

# polars vs pandas feature timings
python3 scripts/benchmark.py --n-days 6000 --tickers 100
```

`fetch_data.py` writes `data_cache/bars.parquet`. Everything downstream reads
that file, so backtests do not touch the network.

Writing a signal:

```python
import numpy as np, polars as pl
from backtester import run_backtest, CostModel
from backtester.signal import BaseSignal

class MyMomentum(BaseSignal):
    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        c = frame["close"].to_numpy()
        pos = np.zeros(len(c))
        pos[60:] = np.sign(c[60:] / c[:-60] - 1.0)
        return pl.Series("position", pos)

res = run_backtest(bars, MyMomentum(), costs=CostModel(1.0, 0.5))
print(res.metrics.sharpe, res.metrics.turnover)
```

A signal returns a target position in `[-1, 1]`, not a trade. Position sizing,
fills, costs and execution lag are handled by the engine.

Any panel with columns `date, ticker, open, high, low, close, adj_close, volume`
works as input:

```python
bars = pl.read_csv("my_bars.csv", try_parse_dates=True)
data.validate_bars(bars)
panel = data.align(bars)
```

## Execution model

A position decided from bar `t` data is executed at bar `t+1`. The engine
applies this shift before positions are multiplied by returns, so a signal
cannot opt out of it:

```python
pl.col("raw_position").shift(lag).over("ticker").fill_null(0.0).alias("position")
```

Default mode is `execution="next_open"`:

| step | timing |
|---|---|
| decision | close of bar `t-1`, using rows `0..t-1` |
| fill | open of bar `t` |
| mark out | open of bar `t+1`, `ret_t = open_{t+1}/open_t - 1` |

The fill price is never a price the signal observed. `execution="close_to_close"`
uses `close_t/close_{t-1} - 1` and carries two bars of lag, since a fill at
`close_{t-1}` requires the decision to predate that close. `extra_lag` adds
further delay; negative values raise.

## Look-ahead controls

**Row-wise signals cannot access future rows.** `Signal.predict(history)`
receives a Polars slice of rows `0..t`. Later bars are not present in the
object. Covered by `test_row_wise_interface_cannot_see_the_future`.

**Vectorised signals are audited before use.** `predict_all` receives the whole
panel and runs roughly 100-1000x faster, so `run_backtest` first runs a
future-perturbation audit: cut the panel at row `k`, replace all bars from `k`
onward, recompute positions, and compare rows `0..k-1`. A causal signal produces
identical values. Any difference raises `LeakageError` naming the row. Passing
`audit=False` skips the check and is recorded in `result.config["audited"]`.

Leak types this detects:

| leak | typical form | perturbation that catches it |
|---|---|---|
| off-by-one | `shift(-1)` returning next row's price | shuffle |
| full-sample statistic | `np.nanquantile(vol, 0.9)` over all rows | scale |
| pre-split scaler fit | `StandardScaler().fit(all_data)` | scale |
| restated field | as-of-today value joined onto historical rows | either |

`PeekingVolFilter` and `VolFilteredMomentum` in `signals.py` differ only in
whether the volatility threshold is a full-sample or expanding quantile. The
audit separates them; reading the diff does not.

**Feature causality.** `features.assert_causal(build, bars)` perturbs the second
half of a panel, rebuilds features, and compares the first half element by
element. `features.lag(expr, k)` raises for `k < 0`.

**Alignment.** `align(..., missing="ffill")` uses `forward_fill` only; there is
no backward fill in the package. The default `missing="null"` marks absent bars
untradable, carrying the position without accruing return or charging cost,
rather than treating a halted name as a flat riskless asset.

**Reporting.** Metrics come from the net equity curve. Gross Sharpe is reported
alongside so the cost gap is visible. `walk_forward` returns only concatenated
out-of-sample segments and refits a new signal instance per fold from a factory.
The default `CostModel` is non-zero.

## Tests

46 tests, no network required.

| test | assertion |
|---|---|
| Perfect foresight | `Oracle`, positioned on the sign of the return it earns, scores Sharpe 16.5 against momentum's 0.74 on the same fixture. A lower figure would mean positions and returns are misaligned. |
| Look-ahead detection | All three broken signals raise `LeakageError`; the engine refuses to run them; all four honest signals pass without a false positive. |
| Zero signal | A flat position gives `turnover == 0.0`, `n_trades == 0`, zero total cost, equity flat at 1.0. Buy-and-hold pays the spread once. |
| Cost monotonicity | Net PnL is non-increasing from 0 to 25bp; cost scales linearly with bps. |
| Known answer | Five hand-built bars: opens `100, 110, 121, 121, 108.9`, returns `[+10%, +10%, 0, -10%]`, positions `[0, 1, 1, 1, 1]`, equity `[1, 1.10, 1.10, 0.99, 0.99]`. |

`lag_sensitivity` reports Sharpe against added execution delay. A signal reading
prices near its own fill loses most of its Sharpe at one extra bar, while a
slower effect decays gradually:

| extra bars of delay | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| `momentum` | -0.03 | 0.03 | 0.01 | 0.01 | 0.09 |
| `peek_ahead_momentum` | 5.52 | -0.48 | -0.37 | -0.30 | -0.25 |

![lag sensitivity](reports/lag_sensitivity.png)

## Results

AAPL, MSFT, XOM, JPM, BAC, KO and SPY, daily bars from 2005-01-03 to 2026-09-10,
5,456 days. Costs 1.5bp all-in, risk-free rate 0.0.

Walk-forward, out-of-sample only, 750-bar rolling train and 125-bar test,
37 folds:

| strategy | OOS Sharpe | gross | CAGR | max DD | turnover |
|---|---|---|---|---|---|
| buy_and_hold | 0.61 | 0.61 | 11.9% | -58.4% | 0.000 |
| mean_reversion | 0.25 | 0.31 | 2.4% | -34.2% | 0.158 |
| vol_filtered_momentum | 0.24 | 0.27 | 2.0% | -26.3% | 0.082 |
| momentum | -0.04 | -0.01 | -1.9% | -47.0% | 0.086 |

No signal beats buy-and-hold on this universe and period. Momentum is negative
out of sample. The two positive signals clear zero by less than the spread
across folds supports, so they are not evidence of an edge.

For comparison, the same code path scores 1.30 on the synthetic fixture, whose
data generator contains a momentum effect by construction.

Buy-and-hold's -58% drawdown is the 2008 decline. The volatility filter reduces
it to -26% and gives up most of the return, which is the trade-off the exposure
and turnover columns are there to show.

![equity](reports/equity.png)

Net Sharpe against transaction cost:

| all-in bps | 0 | 1 | 2 | 5 | 10 | 20 | break-even |
|---|---|---|---|---|---|---|---|
| `mean_reversion` | 0.27 | 0.23 | 0.19 | 0.06 | -0.16 | -0.59 | 6.3 bps |
| `vol_filtered_momentum` | 0.25 | 0.23 | 0.20 | 0.13 | 0.01 | -0.23 | 10.4 bps |
| `momentum` | -0.00 | -0.02 | -0.04 | -0.09 | -0.17 | -0.34 | never positive |

Mean reversion and vol-filtered momentum have similar gross Sharpe, 0.27 and
0.25, but break even at 6.3bp and 10.4bp respectively. The difference is
turnover: 0.158 against 0.082. Gross Sharpe alone does not distinguish them.

![cost sensitivity](reports/cost_sensitivity.png)

Out-of-sample Sharpe at 21, 63, 125 and 252-bar refit intervals: 0.65, 0.62,
0.61, 0.61. Results are stable across refit cadence.

Full tables including per-fold detail: [`reports/results.md`](reports/results.md).

## Polars vs pandas

`data.py` and `features.py` use Polars; the rest uses numpy and pandas. The
benchmark computes the same feature set (momentum, SMA, z-score and realised
volatility at four windows, grouped by ticker) in both, checking the outputs
agree to 1e-9 before timing.

600,000 rows (100 tickers x 6,000 days), median of 3 runs:

| implementation | median (s) | speedup |
|---|---|---|
| pandas `groupby().rolling()` | 1.298 | 1.0x |
| polars eager | 0.296 | 4.4x |
| polars lazy | 0.393 | 3.3x |
| polars `scan_parquet` + lazy | 0.416 | 3.1x |

45,000 rows (15 x 3,000):

| implementation | median (s) | speedup |
|---|---|---|
| pandas | 0.117 | 1.0x |
| polars eager | 0.022 | 5.2x |
| polars lazy | 0.020 | 5.8x |

Grouped rolling windows are where Polars wins. pandas
`groupby().rolling()` materialises a frame per group and pays Python overhead
per group, while Polars evaluates `rolling_mean(20).over("ticker")` as a single
vectorised pass over columnar data. The margin is 3-6x and widens with many
groups and many narrow windows.

A second reason for using Polars here is that `shift(k)` and `rolling_*` are the
only causal primitives available in the expression API, which makes a feature's
time dependence explicit rather than a consequence of index alignment. Several
of the leak types listed above are pandas index-alignment mistakes that are
hard to write by accident in expressions.

Lazy evaluation did not help on this workload. At 600k rows the lazy plan is
slower than eager, 0.393s against 0.296s, because there is no filter or
projection to push down and eager execution already parallelises the windowed
aggregations, so planning is pure overhead. `scan_parquet` is slower again
because it adds a disk read the in-memory version does not do. Lazy is worth
using when the query reads a subset of a large file, for example two years and
three columns out of a twenty-year panel.

The cost of mixing both libraries is a `to_pandas()` conversion at the boundary
and a pyarrow dependency. A rewrite would use Polars throughout; the engine's
`group_by` and `over` operations map onto it directly, and the remaining pull
towards pandas is matplotlib and statsmodels interoperability.

## Data handling

Full policy is documented at the top of `data.py`.

- A bar is stamped with its closing date. Row `t` contains only what was
  observable by that close.
- A missing bar means the instrument did not trade. No forward fill by default.
- `validate_bars` rejects duplicate `(ticker, date)` rows, non-monotonic dates,
  `high < low` and non-positive closes on every input path.

Adjusted prices are not point-in-time. yfinance returns split- and
dividend-adjusted history that is restated whenever a new corporate action
occurs, so an adjusted close downloaded today for 2015 differs from the figure
available in 2015. The harness therefore trades raw OHLC (`auto_adjust=False`)
and carries `adj_close` separately for total-return calculations. Fixing this
properly requires vendor snapshots with an as-of date.

## Limitations

- No protection against survivorship bias. The ticker list is supplied by the
  user, and a list drawn from current index membership selects on survival.
- No protection against overfitting across repeated runs. Walk-forward raises
  the cost of tuning on test data but does not eliminate it.
- Adjusted-price restatement is documented, not solved.
- `audit=False` disables leak detection. The flag is recorded in the result
  config but the check is skippable.
- Daily bars only. Intraday execution, borrow costs, shorting constraints and
  capacity limits are not modelled.

## Layout

```
backtester/
  data.py         loading, alignment, validation, point-in-time policy
  features.py     causal feature expressions, assert_causal
  signal.py       Signal and VectorSignal interfaces
  engine.py       positions to PnL, execution lag
  costs.py        spread, commission, participation-linear impact
  metrics.py      Sharpe, drawdown, turnover, exposure, hit rate
  leakguard.py    future-perturbation audit, lag sensitivity
  walkforward.py  rolling refit, out-of-sample stitching
  signals.py      example signals, including deliberately broken ones
  report.py       tables, figures, sensitivity grids
scripts/
  fetch_data.py   yfinance to parquet cache
  run_report.py   full report to reports/
  benchmark.py    polars vs pandas timings
tests/            46 tests, no network
```
