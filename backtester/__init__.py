"""A signal backtesting harness built to be paranoid about look-ahead bias.

The pipeline is deliberately split into modules with hard boundaries:

    data -> features -> signal -> target positions -> fills -> PnL -> metrics

The boundary between `signal` and `engine` is the important one: a signal
returns a *target position*, never a trade, and it never sees a price it could
not have seen at decision time.
"""

from backtester.engine import BacktestResult, run_backtest
from backtester.costs import CostModel
from backtester.metrics import summarise
from backtester.leakguard import LeakageError, detect_lookahead

__all__ = [
    "run_backtest",
    "BacktestResult",
    "CostModel",
    "summarise",
    "detect_lookahead",
    "LeakageError",
]

__version__ = "0.1.0"
