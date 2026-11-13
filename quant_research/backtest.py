"""Next-session, self-financing portfolio simulator with drift-aware turnover.

Market orders are assumed to execute at the next session's *previous close*
(not an execution simulation). This deliberately simplified assumption avoids
using next-day close prices to place the trade and is documented as a limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .data import validate_prices


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    signal_weights: pd.DataFrame


def simulate_portfolio(
    prices: pd.DataFrame,
    signal_weights: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    benchmark: str = "SPY",
) -> BacktestResult:
    """Assume signal generated AFTER date t close, implemented for t+1 return.

    Current holdings drift with price changes between rebalances. Turnover is
    |target - drifted/pretrade holdings|, charging cost before market return.
    Initial investment from cash counts as 100% one-way turnover.
    """
    prices = validate_prices(prices)
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be nonnegative")
    if benchmark not in prices.columns:
        raise ValueError(f"Missing benchmark {benchmark}")
    if not set(signal_weights.columns).issubset(prices.columns):
        raise ValueError("Weights include unknown asset columns")
    if not signal_weights.index.isin(prices.index).all() or not signal_weights.index.is_unique:
        raise ValueError("Signal dates must occur once on observed price dates")
    if not (np.isfinite(signal_weights.to_numpy()).all() and
            (signal_weights.to_numpy() >= -1e-10).all() and
            np.allclose(signal_weights.sum(axis=1), 1.0)):
        raise ValueError("Signal weights must be finite, nonnegative and sum to one")
    signals = signal_weights.reindex(columns=prices.columns, fill_value=0.0).sort_index()
    daily_returns = prices.pct_change(fill_method=None).fillna(0.0)
    signal_by_execution_date: dict[pd.Timestamp, tuple[pd.Timestamp, np.ndarray]] = {}
    for signal_date, target in signals.iterrows():
        idx = int(prices.index.get_loc(signal_date))
        if idx + 1 < len(prices):
            signal_by_execution_date[prices.index[idx + 1]] = (signal_date, target.to_numpy(float))
    if not signal_by_execution_date:
        raise ValueError("No signal with a subsequent trading day")
    first_execution = min(signal_by_execution_date)
    w = np.zeros(prices.shape[1], dtype=float)
    nav = 1.0
    bench_nav = 1.0
    active = False
    records = []
    trades = []
    for date, asset_row in daily_returns.iterrows():
        r = asset_row.to_numpy(dtype=float)
        turnover = 0.0
        cost_rate = 0.0
        if date in signal_by_execution_date:
            source_date, target = signal_by_execution_date[date]
            turnover = float(np.abs(target - w).sum())
            cost_rate = turnover * transaction_cost_bps / 10000.0
            if cost_rate >= 1:
                raise ValueError("Trading cost exceeds portfolio capital")
            trades.append({"signal_date": source_date, "execution_date": date,
                           "turnover": turnover, "cost_fraction": cost_rate})
            w = target.copy()
            active = True
        gross_return = float(np.dot(w, r)) if active else 0.0
        if gross_return <= -1.0:
            raise ValueError("Non-positive asset/portfolio wealth")
        net_return = (1 - cost_rate) * (1 + gross_return) - 1 if active else 0.0
        nav *= 1 + net_return
        if active:
            w = w * (1 + r) / (1 + gross_return)
        bench_return = float(asset_row[benchmark]) if date >= first_execution else 0.0
        bench_nav *= 1 + bench_return
        records.append({"date": date, "portfolio_return": net_return,
                        "gross_return": gross_return, "benchmark_return": bench_return,
                        "equity": nav, "benchmark_equity": bench_nav,
                        "turnover": turnover, "cost_fraction": cost_rate})
    daily = pd.DataFrame(records).set_index("date")
    return BacktestResult(daily=daily, trades=pd.DataFrame(trades), signal_weights=signals)
