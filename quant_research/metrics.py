import numpy as np
import pandas as pd


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def calculate_cagr(equity_curve: pd.Series) -> float:
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

    if years <= 0:
        return np.nan

    return equity_curve.iloc[-1] ** (1 / years) - 1


def calculate_performance_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    turnover: pd.Series | None = None,
) -> pd.Series:
    total_return = equity_curve.iloc[-1] - 1
    cagr = calculate_cagr(equity_curve)

    volatility = returns.std() * np.sqrt(252)

    if returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    drawdown = calculate_drawdown(equity_curve)
    max_drawdown = drawdown.min()

    if max_drawdown != 0:
        calmar_ratio = cagr / abs(max_drawdown)
    else:
        calmar_ratio = np.nan

    positive_day_ratio = (returns > 0).mean()

    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Positive Day Ratio": positive_day_ratio,
    }

    if turnover is not None:
        metrics["Average Daily Turnover"] = turnover.mean()

    return pd.Series(metrics)