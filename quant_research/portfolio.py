"""Portfolio target weights, implementation costs and self-financing P&L."""

import numpy as np
import pandas as pd

from quant_research.config import (
    TOP_N, MAX_WEIGHT, VOL_LOOKBACK, TRANSACTION_COST, BENCHMARK,
)


def enforce_weight_cap(raw_weights: pd.Series, max_weight: float = MAX_WEIGHT) -> pd.Series:
    """Allocate 100% among assets, with each final weight <= max_weight.

    Redistribute excess mass iteratively rather than clipping and then
    renormalising (which can violate the limit again).
    """
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1].")
    if raw_weights.empty:
        raise ValueError("At least one asset is required.")
    if len(raw_weights) * max_weight < 1 - 1e-10:
        raise ValueError("The position cap is infeasible for the asset count.")

    scores = raw_weights.astype(float)
    if not np.isfinite(scores.to_numpy()).all() or (scores < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    if scores.sum() == 0:
        scores = pd.Series(1.0, index=scores.index)

    result = pd.Series(0.0, index=scores.index)
    remaining = 1.0
    free = scores.index

    while len(free):
        free_scores = scores.loc[free]
        proposed = (
            free_scores / free_scores.sum() * remaining
            if free_scores.sum() > 0
            else pd.Series(remaining / len(free), index=free)
        )
        over = proposed > max_weight + 1e-12
        if not over.any():
            result.loc[free] = proposed
            break
        capped = proposed.index[over]
        result.loc[capped] = max_weight
        remaining -= len(capped) * max_weight
        free = proposed.index[~over]

    if not np.isclose(result.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Portfolio weights do not sum to 1.")
    if (result > max_weight + 1e-9).any():
        raise RuntimeError("Portfolio position cap was breached.")
    return result


def calculate_inverse_vol_weights(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    selected_tickers: list[str],
    lookback: int = VOL_LOOKBACK,
    max_weight: float = MAX_WEIGHT,
) -> pd.Series:
    if not selected_tickers:
        raise ValueError("Cannot allocate with no selected tickers.")
    end_position = prices.index.get_loc(date)
    start_position = max(0, end_position - lookback)
    price_window = prices.iloc[start_position:end_position + 1][selected_tickers]
    returns = price_window.pct_change(fill_method=None).dropna()

    if len(returns) < 2:
        scores = pd.Series(1.0, index=selected_tickers)
    else:
        volatility = returns.std() * np.sqrt(252)
        inverse_vol = 1 / volatility.replace(0, np.nan)
        scores = inverse_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if scores.sum() <= 0:
            scores[:] = 1.0

    return enforce_weight_cap(scores, max_weight)


def build_portfolio_weights(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Targets set at signal close; non-signal rows carry latest target.

    The return simulator below does NOT trade daily to the forward-filled
    targets. It lets actual holdings drift between explicit rebalance dates.
    """
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for date, group in predictions.groupby("date"):
        if date not in weights.index:
            continue
        selected = group.sort_values(
            "predicted_forward_return", ascending=False
        ).head(top_n)
        selected_tickers = selected["ticker"].tolist()
        weights.loc[date, :] = 0.0  # explicitly liquidate deselected assets
        if selected_tickers:
            allocation = calculate_inverse_vol_weights(
                prices=prices, date=date, selected_tickers=selected_tickers
            )
            weights.loc[date, allocation.index] = allocation.values

    return weights.ffill().fillna(0.0)


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Calculate close-to-close P&L with buy-and-hold drift between trades.

    At each date, returns accrue on holdings established at the *previous*
    close. A rebalance signal available at today's close is traded *after*
    today's price move, and its cost is charged today. Between rebalance
    dates weights drift with prices, and no fictitious daily rebalance occurs.

    If explicit dates are omitted, target-weight changes are treated as
    rebalances. Pass all actual signal dates to capture rebalance costs even
    where consecutive target allocations happen to be identical.
    """
    if not prices.index.equals(weights.index) or not prices.columns.equals(weights.columns):
        raise ValueError("prices and weights must have identical indices and columns.")
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be nonnegative.")
    if weights.isna().any().any() or (weights < -1e-10).any().any():
        raise ValueError("Target weights must be finite and nonnegative.")
    if (weights.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("Target portfolio weight must not exceed 100%.")

    returns = prices.pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float)
    target = weights.to_numpy(dtype=float)
    if rebalance_dates is None:
        changed = np.any(~np.isclose(target, np.vstack([np.zeros((1, target.shape[1])), target[:-1]]), atol=1e-12), axis=1)
    else:
        changed = prices.index.isin(pd.DatetimeIndex(rebalance_dates))

    holding = np.zeros(prices.shape[1], dtype=float)
    records = []
    equity = 1.0
    for i, today_return in enumerate(returns):
        gross = float(np.dot(holding, today_return))
        pre_trade = holding * (1 + today_return) / (1 + gross)
        turnover = 0.0
        if changed[i]:
            turnover = float(np.abs(target[i] - pre_trade).sum())
            holding = target[i].copy()
        else:
            holding = pre_trade

        cost = turnover * transaction_cost
        net = gross - cost
        equity *= 1 + net
        records.append((gross, cost, net, turnover, equity))

    return pd.DataFrame(
        records,
        index=prices.index,
        columns=["gross_return", "transaction_cost", "portfolio_return", "turnover", "equity_curve"],
    )


def calculate_benchmark_returns(
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    benchmark_returns = prices[benchmark].pct_change(fill_method=None).fillna(0.0)
    benchmark_equity = (1 + benchmark_returns).cumprod()
    return pd.DataFrame({
        "benchmark_return": benchmark_returns,
        "benchmark_equity": benchmark_equity,
    }, index=prices.index)
