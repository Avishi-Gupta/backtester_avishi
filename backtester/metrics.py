"""Performance metrics, computed from the NET equity curve.

Reported together, always. A single number invites cherry-picking; a Sharpe
without a turnover figure beside it is an unanswered question about whether the
costs were real.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    n_periods: int
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    turnover: float
    exposure: float
    n_trades: int
    avg_cost_bps: float
    gross_sharpe: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def sharpe(returns: np.ndarray, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    """Annualised Sharpe of an excess-return series.

    RISK-FREE ASSUMPTION: ``rf`` is the per-period risk-free rate and defaults
    to 0.0. That is a real assumption, not a neutral one -- in a 5% rates
    regime a long-only equity strategy's Sharpe is overstated by roughly
    0.05/vol. State it whenever you quote the number.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf
    sd = excess.std(ddof=1)
    # Floating point makes std() of a constant series tiny-but-nonzero, which
    # would report a Sharpe of 1e16. A constant series has no risk-adjusted
    # information; call it zero.
    if not np.isfinite(sd) or sd <= 1e-15 * max(1.0, abs(excess.mean())):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    """Like Sharpe but penalising only downside deviation."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf")
    dd = np.sqrt((downside ** 2).mean())
    if dd <= 1e-15:
        return float("inf")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def equity_curve(returns: np.ndarray) -> np.ndarray:
    """Compounded equity from a per-period return series, starting at 1.0."""
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    return np.cumprod(1.0 + r)


def max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough fraction. Returned as a negative number."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=float)
    return eq / np.maximum.accumulate(eq) - 1.0


def hit_rate(returns: np.ndarray) -> float:
    """Fraction of periods with a positive return.

    Nearly meaningless in isolation: a strategy that wins 95% of days and loses
    everything on the other 5% has a wonderful hit rate and no future. Quote it
    only next to Sharpe and max drawdown.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float((r > 0).mean())


def turnover(positions: np.ndarray) -> float:
    """Mean absolute position change per period.

    The number that tells you in advance whether costs will eat the signal:
    annualised cost drag is roughly ``turnover * 252 * all_in_bps``.
    """
    p = np.nan_to_num(np.asarray(positions, dtype=float))
    if p.size == 0:
        return float("nan")
    return float(np.abs(np.diff(p, prepend=0.0)).mean())


def exposure(positions: np.ndarray) -> float:
    """Fraction of periods with a non-zero position."""
    p = np.nan_to_num(np.asarray(positions, dtype=float))
    if p.size == 0:
        return float("nan")
    return float((np.abs(p) > 1e-12).mean())


def summarise(
    net_returns: np.ndarray,
    gross_returns: np.ndarray,
    positions: np.ndarray,
    costs: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    rf: float = 0.0,
) -> Metrics:
    """Everything at once, from the net curve."""
    net = np.nan_to_num(np.asarray(net_returns, dtype=float))
    gross = np.nan_to_num(np.asarray(gross_returns, dtype=float))
    pos = np.nan_to_num(np.asarray(positions, dtype=float))
    cst = np.nan_to_num(np.asarray(costs, dtype=float))

    eq = equity_curve(net)
    n = net.size
    years = n / periods_per_year if periods_per_year else float("nan")
    total = float(eq[-1] - 1.0) if n else float("nan")
    cagr = float(eq[-1] ** (1 / years) - 1.0) if n and years > 0 and eq[-1] > 0 else float("nan")

    traded = np.abs(np.diff(pos, prepend=0.0))
    traded_total = traded.sum()

    return Metrics(
        n_periods=n,
        total_return=total,
        cagr=cagr,
        ann_vol=float(net.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan"),
        sharpe=sharpe(net, periods_per_year, rf),
        sortino=sortino(net, periods_per_year, rf),
        max_drawdown=max_drawdown(eq),
        calmar=(
            float(cagr / abs(max_drawdown(eq)))
            if n and max_drawdown(eq) < 0 and np.isfinite(cagr)
            else float("nan")
        ),
        hit_rate=hit_rate(net),
        turnover=turnover(pos),
        exposure=exposure(pos),
        n_trades=int((traded > 1e-12).sum()),
        avg_cost_bps=float(cst.sum() / traded_total * 1e4) if traded_total > 0 else 0.0,
        gross_sharpe=sharpe(gross, periods_per_year, rf),
    )


def metrics_table(named: dict[str, Metrics]) -> pl.DataFrame:
    """Stack several Metrics into one comparison table."""
    rows = []
    for name, m in named.items():
        d = {"strategy": name}
        d.update(m.to_dict())
        rows.append(d)
    return pl.DataFrame(rows)
