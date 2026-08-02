import numpy as np
import pandas as pd

from quant_research.config import (
    TOP_N,
    MAX_WEIGHT,
    VOL_LOOKBACK,
    TRANSACTION_COST,
    BENCHMARK,
)


def calculate_inverse_vol_weights(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    selected_tickers: list[str],
    lookback: int = VOL_LOOKBACK,
    max_weight: float = MAX_WEIGHT,
) -> pd.Series:
    end_position = prices.index.get_loc(date)
    start_position = max(0, end_position - lookback)

    price_window = prices.iloc[start_position:end_position + 1][selected_tickers]
    returns = price_window.pct_change().dropna()

    if returns.empty:
        raw_weights = pd.Series(1 / len(selected_tickers), index=selected_tickers)
    else:
        volatility = returns.std() * np.sqrt(252)
        inv_vol = 1 / volatility.replace(0, np.nan)
        raw_weights = inv_vol / inv_vol.sum()

    raw_weights = raw_weights.fillna(1 / len(selected_tickers))
    capped_weights = raw_weights.clip(upper=max_weight)
    final_weights = capped_weights / capped_weights.sum()

    return final_weights


def build_portfolio_weights(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    weights = pd.DataFrame(
        np.nan,
        index=prices.index,
        columns=prices.columns,
    )

    for date, group in predictions.groupby("date"):
        if date not in weights.index:
            continue

        selected = (
            group
            .sort_values("predicted_forward_return", ascending=False)
            .head(top_n)
        )

        selected_tickers = selected["ticker"].tolist()

        weights.loc[date, :] = 0.0

        if len(selected_tickers) > 0:
            allocation = calculate_inverse_vol_weights(
                prices=prices,
                date=date,
                selected_tickers=selected_tickers,
            )

            weights.loc[date, allocation.index] = allocation.values

    weights = weights.ffill()
    weights = weights.fillna(0.0)

    return weights


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    asset_returns = prices.pct_change().fillna(0)

    shifted_weights = weights.shift(1).fillna(0)
    gross_returns = (shifted_weights * asset_returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    costs = turnover * transaction_cost

    net_returns = gross_returns - costs
    equity_curve = (1 + net_returns).cumprod()

    result = pd.DataFrame(index=prices.index)
    result["gross_return"] = gross_returns
    result["transaction_cost"] = costs
    result["portfolio_return"] = net_returns
    result["turnover"] = turnover
    result["equity_curve"] = equity_curve

    return result


def calculate_benchmark_returns(
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    benchmark_returns = prices[benchmark].pct_change().fillna(0)
    benchmark_equity = (1 + benchmark_returns).cumprod()

    result = pd.DataFrame(index=prices.index)
    result["benchmark_return"] = benchmark_returns
    result["benchmark_equity"] = benchmark_equity

    return result