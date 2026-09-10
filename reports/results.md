# Results

- data: `synthetic (seed 7, 3000 bars)`
- span: 2005-01-03 to 2016-07-01, 4 tickers, 3,000 bars
- costs: 1.50bp linear
- execution: decide at close t-1, fill at open t, mark out at open t+1
- risk-free rate assumed 0.0 in all Sharpe figures

## Full-sample summary (in-sample; NOT the result)

Shown for completeness and as a sanity check on the plumbing. The
number to quote is the walk-forward one below.

| strategy | sharpe | gross_sharpe | cagr | ann_vol | max_drawdown | calmar | turnover | exposure | hit_rate | n_trades | avg_cost_bps | n_periods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | 0.480 | 0.480 | 0.052 | 0.122 | -0.677 | 0.077 | 0.000 | 1.000 | 0.504 | 1 | 1.500 | 3000 |
| momentum | 1.116 | 1.148 | 0.120 | 0.106 | -0.332 | 0.361 | 0.069 | 0.980 | 0.516 | 2939 | 2.005 | 3000 |
| mean_reversion | -1.139 | -1.031 | -0.086 | 0.076 | -0.658 | -0.130 | 0.116 | 0.993 | 0.469 | 2979 | 2.794 | 3000 |
| vol_filtered_momentum | 1.364 | 1.401 | 0.131 | 0.093 | -0.255 | 0.513 | 0.070 | 0.980 | 0.526 | 2939 | 1.941 | 3000 |

## Walk-forward, out-of-sample only

| strategy | folds | oos_sharpe | oos_gross_sharpe | oos_cagr | oos_max_dd | turnover | exposure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | 18 | 0.309 | 0.309 | 0.031 | -0.677 | 0.000 | 1.000 |
| momentum | 18 | 1.100 | 1.134 | 0.119 | -0.319 | 0.072 | 1.000 |
| mean_reversion | 18 | -1.075 | -0.967 | -0.081 | -0.561 | 0.117 | 1.000 |
| vol_filtered_momentum | 18 | 1.300 | 1.338 | 0.125 | -0.248 | 0.073 | 1.000 |

Per-fold detail for `vol_filtered_momentum` (the spread across folds matters more than the mean):

| fold | train_bars | test_bars | sharpe | max_drawdown | turnover |
| --- | --- | --- | --- | --- | --- |
| 0 | 750 | 125 | 0.873 | -0.064 | 0.093 |
| 1 | 750 | 125 | 7.378 | -0.019 | 0.039 |
| 2 | 750 | 125 | 7.489 | -0.024 | 0.034 |
| 3 | 750 | 125 | 2.759 | -0.043 | 0.067 |
| 4 | 750 | 125 | 2.170 | -0.032 | 0.077 |
| 5 | 750 | 125 | 2.094 | -0.033 | 0.054 |
| 6 | 750 | 125 | -0.598 | -0.052 | 0.088 |
| 7 | 750 | 125 | 0.481 | -0.040 | 0.061 |
| 8 | 750 | 125 | -0.061 | -0.063 | 0.101 |
| 9 | 750 | 125 | 1.431 | -0.028 | 0.063 |
| 10 | 750 | 125 | -1.069 | -0.066 | 0.099 |
| 11 | 750 | 125 | 0.625 | -0.071 | 0.081 |
| 12 | 750 | 125 | 0.703 | -0.070 | 0.099 |
| 13 | 750 | 125 | -0.667 | -0.100 | 0.076 |
| 14 | 750 | 125 | 0.163 | -0.047 | 0.083 |
| 15 | 750 | 125 | 1.572 | -0.063 | 0.079 |
| 16 | 750 | 125 | -1.826 | -0.172 | 0.075 |
| 17 | 750 | 125 | -2.479 | -0.117 | 0.094 |

## Cost sensitivity

### momentum

| all_in_bps | sharpe_net | cagr | max_drawdown | turnover | annual_cost_drag |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 1.148 | 0.124 | -0.324 | 0.069 | 0.000 |
| 1.000 | 1.126 | 0.121 | -0.329 | 0.069 | 0.002 |
| 2.000 | 1.105 | 0.118 | -0.334 | 0.069 | 0.003 |
| 5.000 | 1.039 | 0.111 | -0.349 | 0.069 | 0.009 |
| 10.000 | 0.929 | 0.098 | -0.372 | 0.069 | 0.017 |
| 20.000 | 0.710 | 0.072 | -0.417 | 0.069 | 0.035 |

Break-even all-in cost: beyond the tested grid.

### mean_reversion

| all_in_bps | sharpe_net | cagr | max_drawdown | turnover | annual_cost_drag |
| --- | --- | --- | --- | --- | --- |
| 0.000 | -1.031 | -0.078 | -0.625 | 0.116 | 0.000 |
| 1.000 | -1.103 | -0.083 | -0.647 | 0.116 | 0.003 |
| 2.000 | -1.175 | -0.088 | -0.668 | 0.116 | 0.006 |
| 5.000 | -1.390 | -0.103 | -0.725 | 0.116 | 0.015 |
| 10.000 | -1.749 | -0.127 | -0.801 | 0.116 | 0.029 |
| 20.000 | -2.467 | -0.173 | -0.896 | 0.116 | 0.059 |

Break-even all-in cost: beyond the tested grid.

### vol_filtered_momentum

| all_in_bps | sharpe_net | cagr | max_drawdown | turnover | annual_cost_drag |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 1.401 | 0.135 | -0.252 | 0.070 | 0.000 |
| 1.000 | 1.376 | 0.132 | -0.254 | 0.070 | 0.002 |
| 2.000 | 1.352 | 0.130 | -0.256 | 0.070 | 0.004 |
| 5.000 | 1.278 | 0.122 | -0.263 | 0.070 | 0.009 |
| 10.000 | 1.156 | 0.109 | -0.279 | 0.070 | 0.018 |
| 20.000 | 0.910 | 0.084 | -0.361 | 0.070 | 0.035 |

Break-even all-in cost: beyond the tested grid.

## Latency sensitivity

Extra bars of delay ON TOP of the mandatory one. A signal whose Sharpe
falls off a cliff between 0 and 1 was reading something close to its own
fill price. `peek_ahead_momentum` is included as the positive control.

### momentum

| extra_lag_bars | sharpe_net | sharpe_gross | max_drawdown |
| --- | --- | --- | --- |
| 0 | 1.116 | 1.148 | -0.332 |
| 1 | 1.071 | 1.104 | -0.287 |
| 2 | 1.139 | 1.172 | -0.240 |
| 3 | 1.054 | 1.087 | -0.227 |
| 5 | 1.108 | 1.141 | -0.317 |

### mean_reversion

| extra_lag_bars | sharpe_net | sharpe_gross | max_drawdown |
| --- | --- | --- | --- |
| 0 | -1.139 | -1.031 | -0.658 |
| 1 | -1.021 | -0.915 | -0.638 |
| 2 | -1.051 | -0.946 | -0.640 |
| 3 | -0.932 | -0.825 | -0.604 |
| 5 | -0.986 | -0.881 | -0.665 |

### peek_ahead_momentum

| extra_lag_bars | sharpe_net | sharpe_gross | max_drawdown |
| --- | --- | --- | --- |
| 0 | 11.793 | 11.930 | -0.032 |
| 1 | 0.811 | 0.950 | -0.170 |
| 2 | 0.227 | 0.364 | -0.423 |
| 3 | 0.388 | 0.524 | -0.378 |
| 5 | 0.792 | 0.926 | -0.260 |

## Refit-cadence sensitivity

Out-of-sample Sharpe for `vol_filtered_momentum` as the refit interval changes. Flat is
good news; a single lucky cadence is not a result.

| test_bars | n_folds | oos_sharpe | oos_max_dd | turnover |
| --- | --- | --- | --- | --- |
| 21 | 107 | 1.285 | -0.253 | 0.073 |
| 63 | 35 | 1.387 | -0.196 | 0.072 |
| 125 | 18 | 1.300 | -0.248 | 0.073 |
| 252 | 8 | 1.820 | -0.109 | 0.071 |

## Figures

![equity](equity.png)

![cost sensitivity](cost_sensitivity.png)

![lag sensitivity](lag_sensitivity.png)
