# A signal backtesting harness

A **tool, not a strategy**. You hand it a signal function; it hands you back an
honest evaluation of that signal, or refuses to evaluate it at all.

The design premise is one specific failure: a model that scored well because its
evaluation was contaminated. Look-ahead bias is the same failure wearing market
clothes — the information used to make a decision was not available when the
decision was made. Everything structural in this repo exists to make that
failure either impossible to express or impossible to hide.

```
raw data -> features -> signal -> target positions -> fills -> PnL -> metrics
```

Each stage is a module with a hard boundary. The signal cannot reach into raw
data, cannot choose its own fill price, and cannot decide its own latency.

---

## The one rule

> **A decision made using information from bar `t` can only be executed at bar `t+1`.**

Enforced in `engine.py`, in one line, before any return is ever multiplied by a
position:

```python
pl.col("raw_position").shift(lag).over("ticker").fill_null(0.0).alias("position")
```

The rule lives in the framework rather than in user code because a rule that
lives in user code is a rule that gets forgotten in the third notebook. There is
no argument that removes the shift; `extra_lag` can only *add* to it, and a
negative value raises.

**Timeline** (default `execution="next_open"`):

| when | what happens |
|---|---|
| close of bar `t-1` | signal sees rows `0..t-1`, returns a target position in `[-1, 1]` |
| open of bar `t` | the engine fills that target — a price the signal never saw |
| open of bar `t+1` | the position is marked out; `ret_t = open_{t+1}/open_t - 1` |

`execution="close_to_close"` is also available and carries **two** bars of
latency, because filling at `close_{t-1}` on a decision taken at that same close
is the classic fantasy fill.

---

## What this harness refuses to let you do

The section a quant researcher reads first.

**1. It refuses to let a row-wise signal see the future at all.**
`Signal.predict(history)` receives a Polars slice containing rows `0..t`. There
is no `t+1` in the object. Look-ahead here is not detected — it is
*unrepresentable*. `test_row_wise_interface_cannot_see_the_future` asserts the
engine hands over exactly `t+1` rows at step `t`, never more.

**2. It refuses to trust a vectorised signal that has not been audited.**
The fast interface (`predict_all`, ~100–1000x quicker) hands you the whole
panel, which buys back the ability to cheat. So `run_backtest` runs a
**future-perturbation audit** before it will use the output: take the panel, cut
it at `k`, replace every bar at or after `k` with different numbers, ask for the
positions again, and compare the first `k`. A causal signal cannot have moved
them. If one moved, the audit names the row and raises `LeakageError`. Turning
the audit off is allowed and is recorded in `result.config["audited"]`, so the
report can say out loud that it was off.

This catches things a code review does not:

| leak | how it usually gets written | caught by |
|---|---|---|
| off-by-one | `shift(-1)` meaning "next row's feature", getting next row's price | `shuffle` perturbation, instantly |
| full-sample statistic | `np.nanquantile(vol, 0.9)` once, over everything | `scale` perturbation |
| fitted-then-split scaler | `StandardScaler().fit(all_data)` before the split | `scale` perturbation |
| restated field | joining an as-of-today value back onto old rows | either |

`PeekingVolFilter` in `signals.py` is the second row of that table, and it is
four characters different from the honest `VolFilteredMomentum`
(`np.nanquantile(vol, q)` vs an expanding quantile). The audit tells them apart;
a diff does not.

**3. It refuses to let a signal choose its own execution price or latency.**
A signal returns a *target position*, never a trade. Sizing, netting, costs and
the `t+1` shift belong to the engine, where they can be tested.

**4. It refuses to fill a missing bar from the future.**
`align(..., missing="ffill")` uses `forward_fill` only. There is no backward
fill anywhere in the package. The default `missing="null"` treats an absent bar
as untradable — the position is carried, nothing accrues, nothing is charged —
rather than turning a halted or delisted name into a flat, riskless, tradable
asset. `test_alignment_does_not_fill_from_the_future` pins this down.

**5. It refuses to let a feature reach forward.**
`features.assert_causal(build, bars)` scrambles the future half of the panel,
rebuilds, and compares the past half value by value. `features.lag(expr, k)`
raises on `k < 0`.

**6. It refuses to report gross as a result.**
Every metric is computed from the **net** equity curve. Gross Sharpe is carried
alongside, always, so the gap is visible rather than optional. The default
`CostModel` is *not* zero.

**7. It refuses to hand back an in-sample curve as the answer.**
`walk_forward` returns only the concatenated out-of-sample segments. The
in-sample curve is not returned, because it is not a result. Each fold gets a
**fresh** signal from a factory, because reusing one instance across folds is
leakage wearing a different hat.

**8. It refuses to quote a Sharpe on its own.**
`summarise` returns net Sharpe, gross Sharpe, CAGR, vol, max drawdown, Calmar,
turnover, exposure, hit rate, trade count and realised cost in bps as one
object. Hit rate's docstring says in as many words that it is nearly meaningless
alone.

**What it does not protect you against.** Survivorship bias — the ticker list is
yours, and if you picked it from today's index membership no framework can
rescue you. Regime-specific overfitting across many re-runs. Restatement of
adjusted history (documented, not solved). And the fact that a signal author who
really wants to cheat can always pass `audit=False`; the harness makes that a
recorded choice, not a silent one.

---

## The five tests that carry the credibility

Run: `pytest -q` → **46 passed in ~2s**, no network required.

| # | test | what it proves |
|---|---|---|
| 1 | **Perfect foresight** | `Oracle`, which knows the sign of the return it will earn, scores **Sharpe 16.5** vs momentum's 0.74 on the same single-ticker fixture. If it did not, the engine misaligns positions and returns by a bar, and *every* honest result is silently wrong too. |
| 2 | **Look-ahead detection** | All three broken signals are caught by `detect_lookahead`; the engine refuses to run them; and all four honest signals pass without a false positive (a guard that cries wolf is a guard people switch off). |
| 3 | **Zero signal** | A constant flat position gives exactly `turnover == 0.0`, `n_trades == 0`, `cost.sum() == 0.0`, and equity flat at 1.0. Buy-and-hold pays the spread exactly once. |
| 4 | **Cost monotonicity** | Net PnL is non-increasing across 0 → 25bp for every strategy, and cost scales exactly linearly with bps. |
| 5 | **Known answer** | A hand-built five-bar panel: opens `100, 110, 121, 121, 108.9`, returns `[+10%, +10%, 0, −10%]`, held `[0, 1, 1, 1, 1]`, equity `[1, 1.10, 1.10, 0.99, 0.99]`. Verified by hand in the docstring — including the fact that a round trip back to the start price *loses* 1%, because compounding is not additive. |

Plus a sixth diagnostic that is really the point of the whole thing:

**Latency decay.** `lag_sensitivity` reports Sharpe against extra bars of delay.

| extra bars of delay | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| `momentum` (honest) | 1.12 | 1.07 | 1.14 | 1.05 | 1.11 |
| `peek_ahead_momentum` (leaking) | **11.79** | 0.81 | 0.23 | 0.39 | 0.79 |

A real effect decays gently. An edge that is really tomorrow's price falls off a
cliff between 0 and 1. That shape is the signature, and it is why the diagnostic
is worth running on anything that looks too good.

![lag sensitivity](reports/lag_sensitivity.png)

---

## Results

**Real data.** 7 US large caps plus SPY (AAPL, MSFT, XOM, JPM, BAC, KO, SPY),
daily bars 2005-01-03 to 2026-09-10, 5,456 days. Costs 1.5bp all-in, risk-free
rate assumed 0.0. Reproduce with the two commands under *Using real data*. The
deterministic synthetic fixture is still what the tests run on, so `pytest`
needs no network.

**Walk-forward, out-of-sample only** (750-bar rolling train, 125-bar test, 37 folds):

| strategy | OOS Sharpe | gross | CAGR | max DD | turnover |
|---|---|---|---|---|---|
| **buy_and_hold** | **0.61** | 0.61 | 11.9% | −58.4% | 0.000 |
| mean_reversion | 0.25 | 0.31 | 2.4% | −34.2% | 0.158 |
| vol_filtered_momentum | 0.24 | 0.27 | 2.0% | −26.3% | 0.082 |
| momentum | −0.04 | −0.01 | −1.9% | −47.0% | 0.086 |

**None of the signals beat buy-and-hold.** That is the result, and it is
reported as the result. Over a 21-year sample dominated by an equity bull
market, a long-only benchmark with zero turnover earned 0.61 while a
cross-sectional momentum rule earned less than nothing. The two signals that
are positive are positive by a margin that a 37-fold spread does not support.

The honest reading is narrower still: on this universe and this period, the
harness says *there is no edge here* — and a backtester whose value depends on
finding one would have been tuned until it did. What the tool demonstrates is
that it can tell the difference. The synthetic fixture, where a genuine
momentum effect was built into the data generator, scores 1.30 on the same
code path.

Note the drawdowns. Buy-and-hold's −58% is 2008 arriving in full; the
vol-filtered signal cuts that to −26% while giving up most of the return, which
is the trade the `exposure` and `turnover` columns exist to make visible.

![equity](reports/equity.png)

**Cost sensitivity** — net Sharpe as all-in cost rises:

| all-in bps | 0 | 1 | 2 | 5 | 10 | 20 | break-even |
|---|---|---|---|---|---|---|---|
| `mean_reversion` | 0.27 | 0.23 | 0.19 | 0.06 | −0.16 | −0.59 | **6.3 bps** |
| `vol_filtered_momentum` | 0.25 | 0.23 | 0.20 | 0.13 | 0.01 | −0.23 | **10.4 bps** |
| `momentum` | −0.00 | −0.02 | −0.04 | −0.09 | −0.17 | −0.34 | never positive |

Mean reversion dies at 6.3bp. Its gross Sharpe of 0.27 is the more flattering
number and it is 39% higher than the net figure, entirely because turnover is
0.156 — nearly twice the vol-filtered signal's. Two strategies with almost
identical gross Sharpe, and one of them is half as viable. That is the whole
argument for reporting turnover next to Sharpe rather than underneath it.

![cost sensitivity](reports/cost_sensitivity.png)

**Refit-cadence sensitivity** — OOS Sharpe at 21/63/125/252-bar refit intervals:
0.65 / 0.62 / 0.61 / 0.61. Flat, which is the good news you want; the result
does not depend on a cadence nobody thought of as a parameter.

**The leak detector on real data.** `peek_ahead_momentum` — momentum computed
with a window that includes tomorrow's close — scores **5.52** at zero extra
latency and **−0.48** with one extra bar. Gross to net barely moves it; one bar
of honest delay annihilates it. Every genuine signal in the table above is flat
in that same column.

| extra bars of delay | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| `peek_ahead_momentum` (leaking) | **5.52** | −0.48 | −0.37 | −0.30 | −0.25 |
| `momentum` (honest) | −0.03 | 0.03 | 0.01 | 0.01 | 0.09 |

Full tables, including per-fold detail: [`reports/results.md`](reports/results.md).

---

## Polars vs pandas

`data.py` and `features.py` are Polars; the rest is numpy/pandas. Same feature
set (momentum, SMA, z-score, realised vol at 4 windows, grouped by ticker),
verified to agree to 1e-9 before timing.

**600,000 rows (100 tickers × 6,000 days), median of 3 runs:**

| implementation | median (s) | speedup |
|---|---|---|
| pandas `groupby().rolling()` | 1.298 | 1.0x |
| polars eager | 0.296 | **4.4x** |
| polars lazy | 0.393 | 3.3x |
| polars `scan_parquet` → lazy | 0.416 | 3.1x |

**45,000 rows (15 × 3,000):**

| implementation | median (s) | speedup |
|---|---|---|
| pandas | 0.117 | 1.0x |
| polars eager | 0.022 | 5.2x |
| polars lazy | 0.020 | **5.8x** |

### The opinion, which is the actual deliverable here

**Where Polars helped.** Grouped rolling windows, decisively. pandas
`groupby().rolling()` materialises a group per key and pays Python overhead per
group; Polars runs `rolling_mean(20).over("ticker")` as one vectorised Rust pass
over a columnar buffer. 3–6x, consistently, and the gap is widest where the
workload is many narrow windows over many groups — which is precisely what
feature engineering is.

**Where Polars helped for a reason that is not speed.** `shift(k)` and
`rolling_*` are the *only* causal primitives, and they read as such. The
expression API makes "what does this row depend on" a syntactic property rather
than something you infer from index alignment. Half the leak classes in the
table above are pandas index-alignment accidents that the expression API cannot
express by accident. That mattered more to this project than the 4x.

**Where lazy did not help, and I expected it to.** At 600k rows the lazy plan is
*slower* than eager (0.393 vs 0.296). The optimiser has nothing to push down —
no filter, no projection, every column is used — so all it adds is planning
overhead, and eager already parallelises the `.over()` windows. `scan_parquet`
is slower still, because reading 600k rows from disk is real work that the
in-memory version skips. **Lazy pays off when there is something to optimise
away**: scanning a 20-year parquet panel and using three columns and two years
of it. It is not a free win, and I would not use it for the benchmark's
workload.

**Where the split gets awkward.** Two dataframe libraries in one codebase means
`to_pandas()` at the boundary and a pyarrow dependency you did not ask for. If I
were starting again I would go Polars end-to-end; the engine's `group_by`/`over`
work is a natural fit, and the only real pull towards pandas is matplotlib and
statsmodels interop.

Reproduce: `python scripts/benchmark.py --n-days 6000 --tickers 100`.

---

## Data and the point-in-time policy

Written out in full at the top of `data.py`. The short version:

- A bar is stamped with the date it **closed**; everything in row `t` was
  observable by that close.
- Missing bar = "not trading". No forward fill by default. Opt in with
  `missing="ffill"` and own the assertion that the price did not move.
- **Adjusted prices are not point-in-time.** yfinance returns split- and
  dividend-adjusted history, and that history is *restated* whenever a new
  corporate action occurs. The 2015 adjusted close you download today is not a
  number anyone could have traded on in 2015. So the harness trades on **raw
  OHLC** (`auto_adjust=False`) and carries `adj_close` alongside for total-return
  work only. This is a real limitation, not a solved problem: a genuine
  point-in-time solution needs vendor snapshots with an as-of date, which
  yfinance does not provide.
- `validate_bars` rejects duplicate `(ticker, date)` rows, non-monotonic dates,
  `high < low`, and non-positive closes at the door, because each of those
  silently corrupts a backtest rather than crashing it.

---

## Using real data

The synthetic fixture exists so tests and CI run offline. For anything you would
show someone, point the same scripts at real bars.

```bash
python3 -m pip install -e ".[data]"
python3 scripts/fetch_data.py --tickers AAPL MSFT XOM JPM BAC KO SPY --start 2005-01-01
python3 scripts/run_report.py --bars data_cache/bars.parquet
```

Use `python3 -m pip`, not a bare `pip`. A bare `pip` frequently belongs to a
different environment than the `python3` that then runs the script, which is how
a package installs successfully and still imports as missing. The scripts check
their own dependencies on startup and print this command rather than a bare
`ModuleNotFoundError`. They also add the repo root to `sys.path` themselves, so
`python3 scripts/...` works from any directory even without the editable
install.

`fetch_data.py` downloads once and writes `data_cache/bars.parquet`. Everything
downstream reads that cache, so a backtest never touches the network and a
result is reproducible from a file you can inspect. Re-run the fetch only when
you deliberately want fresh data — and expect your numbers to move, for the
restatement reason below.

**Any panel works, not just yfinance.** The engine only needs a long Polars
frame with `date, ticker, open, high, low, close, adj_close, volume`. A CSV from
anywhere becomes a valid input in four lines:

```python
import polars as pl
from backtester import data

bars = pl.read_csv("my_bars.csv", try_parse_dates=True)
data.validate_bars(bars)          # rejects dupes, gaps in ordering, high < low
panel = data.align(bars)          # union calendar; missing bar = untradable
```

`validate_bars` runs on every path in and rejects duplicate `(ticker, date)`
rows, non-monotonic dates, `high < low` and non-positive closes — each of which
corrupts a backtest silently rather than crashing it.

### Three things that change when the data is real

**Pick the ticker list before you look at returns.** The harness cannot save you
from survivorship bias: if the names come from today's index membership, you
have selected on having survived, and every result is flattered. Write down why
each ticker is in the list, in the README, before running anything.

**Adjusted history gets restated.** yfinance returns split- and
dividend-adjusted prices, and those are recomputed whenever a new corporate
action lands, so today's 2015 adjusted close is not a number anyone could have
traded in 2015. The harness trades raw OHLC (`auto_adjust=False`) and keeps
`adj_close` beside it for total-return work only. This is documented, not
solved — a real fix needs vendor snapshots with an as-of date.

**Real calendars have holes.** Different tickers halt, list late and delist. The
default `missing="null"` treats an absent bar as untradable (position carried,
nothing accrued, nothing charged) rather than as a flat riskless day. If you
switch to `missing="ffill"`, you are asserting the price did not move — say so
in the writeup.

Then re-run the audit on your own signals against the real panel. Leaks that
hide in smooth synthetic data surface fast on real prices, where corporate
actions and gaps give a peeking feature much more to grab onto.

---

## Layout

```
backtester/
  data.py         load, align, validate; the point-in-time policy in prose
  features.py     causal feature expressions + assert_causal
  signal.py       the two interfaces, and why there are two
  engine.py       positions -> fills -> PnL; the t+1 rule
  costs.py        spread / commission / participation-linear impact
  metrics.py      Sharpe, drawdown, turnover, exposure, hit rate
  leakguard.py    future-perturbation audit; lag sensitivity
  walkforward.py  rolling refit; out-of-sample stitching
  signals.py      3 honest signals, 3 deliberately broken ones
  report.py       tables, figures, cost/lag sensitivity
scripts/
  fetch_data.py   yfinance -> parquet cache (run once)
  run_report.py   full report -> reports/
  benchmark.py    polars vs pandas, with a correctness check
tests/            46 tests, no network
```

## Quickstart

```bash
cd backtester
python3 -m pip install -e ".[data,dev]"      # deps come from pyproject.toml

pytest -q                                    # 46 tests, offline, ~2s
python scripts/run_report.py                 # synthetic fixture
python scripts/fetch_data.py --tickers AAPL MSFT XOM JPM BAC KO SPY --start 2005-01-01
python scripts/run_report.py --bars data_cache/bars.parquet
python scripts/benchmark.py --n-days 6000 --tickers 100
```

Writing a signal:

```python
from backtester import run_backtest, CostModel
from backtester.signal import BaseSignal
import numpy as np, polars as pl

class MyMomentum(BaseSignal):
    def predict_all(self, frame: pl.DataFrame) -> pl.Series:
        c = frame["close"].to_numpy()
        pos = np.zeros(len(c))
        pos[60:] = np.sign(c[60:] / c[:-60] - 1.0)   # only rows <= t
        return pl.Series("position", pos)            # target position in [-1, 1]

res = run_backtest(bars, MyMomentum(), costs=CostModel(1.0, 0.5))
print(res.metrics.sharpe, res.metrics.turnover)
```

If `MyMomentum` peeks, `run_backtest` raises `LeakageError` naming the row —
before it computes a single number of PnL.
