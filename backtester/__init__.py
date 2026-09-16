"""Daily-bar backtesting harness for evaluating trading signals.

The pipeline is split into modules with fixed boundaries:

    data -> features -> signal -> target positions -> fills -> PnL -> metrics

A signal returns a target position and never a trade, so position sizing,
execution lag and transaction costs stay in the engine.
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
