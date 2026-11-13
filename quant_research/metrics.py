"""Standard performance statistics; report units as decimal rates."""
import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax().clip(lower=np.finfo(float).tiny)
    return float((equity / peak - 1).min())


def summarize(returns: pd.Series, equity: pd.Series, annualization: int = 252) -> dict[str, float]:
    if not len(returns) or len(returns) != len(equity):
        raise ValueError("Missing or inconsistent performance series")
    if not np.isfinite(returns).all() or not np.isfinite(equity).all():
        raise ValueError("Non-finite returns or equity")
    n = int(len(returns))
    years = n / annualization
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else float("nan")
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if n > 1 else 0.0
    sharpe = float(returns.mean() * annualization / volatility) if volatility > 0 else float("nan")
    return {"CAGR": cagr, "annualized_volatility": volatility, "Sharpe_rf0": sharpe,
            "max_drawdown": max_drawdown(equity), "total_return": float(equity.iloc[-1] - 1)}
