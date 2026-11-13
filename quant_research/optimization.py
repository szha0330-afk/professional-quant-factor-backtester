"""21-day mean-variance allocation with analytic multivariable derivatives.

Uses only data through the signal date. Intended as an alternative to the
existing inverse-volatility allocation, not a replacement for the baseline.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def objective(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
              risk_aversion: float) -> float:
    """Minimize risk_aversion/2 * w.T Sigma w - mu.T w."""
    return float(0.5 * risk_aversion * (w @ sigma @ w) - mu @ w)


def gradient(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
             risk_aversion: float) -> np.ndarray:
    """Analytic multivariable gradient."""
    return risk_aversion * (sigma @ w) - mu


def hessian(sigma: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Analytic matrix of second partial derivatives."""
    return risk_aversion * sigma


def variance_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Asset contributions to portfolio variance, summing to w.T Sigma w."""
    return w * (sigma @ w)


def optimize_mean_variance(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    selected_tickers: list[str],
    predicted_returns: pd.Series,
    lookback: int = 63,
    horizon: int = 21,
    max_weight: float = 0.5,
    risk_aversion: float = 8.0,
    shrinkage: float = 0.1,
) -> pd.Series:
    """Use past daily returns and same-date forward-return forecasts.

    Forecasts (mu) cover the next `horizon` trading days. Historical daily
    covariance is scaled by that horizon; this assumes negligible serial
    correlation and is an approximation. 10% diagonal shrinkage stabilizes
    the covariance estimate for small samples.
    """
    if not selected_tickers:
        raise ValueError('At least one selected asset is required.')
    n = len(selected_tickers)
    if not 0 < max_weight <= 1 or n * max_weight < 1 - 1e-10:
        raise ValueError('Full investment is infeasible under this weight cap.')
    if lookback < 30 or horizon <= 0 or risk_aversion <= 0:
        raise ValueError('Invalid lookback, horizon, or risk-aversion parameter.')
    if not 0 <= shrinkage <= 1:
        raise ValueError('shrinkage must be in [0, 1].')
    if date not in prices.index:
        raise ValueError('Signal date must be in prices.index.')
    if len(set(selected_tickers)) != n:
        raise ValueError('Duplicate tickers are not supported.')

    price_window = prices.loc[:date, selected_tickers].tail(lookback + 1)
    daily_returns = price_window.pct_change(fill_method=None).dropna()
    if len(daily_returns) < 30:
        raise ValueError('At least 30 complete historical daily returns required.')

    mu = predicted_returns.reindex(selected_tickers).to_numpy(dtype=float)
    if not np.isfinite(mu).all():
        raise ValueError('Forecast returns must be finite for all selected assets.')

    sigma_daily = daily_returns.cov().to_numpy(dtype=float)
    sigma = horizon * sigma_daily
    sigma = (1 - shrinkage) * sigma + shrinkage * np.diag(np.diag(sigma))
    sigma = 0.5 * (sigma + sigma.T) + 1e-10 * np.eye(n)
    if not np.isfinite(sigma).all():
        raise ValueError('Covariance estimate contains non-finite values.')

    initial_w = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        initial_w,
        args=(mu, sigma, risk_aversion),
        method='SLSQP',
        jac=gradient,
        bounds=[(0.0, max_weight)] * n,
        constraints=[{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
            'jac': lambda w: np.ones_like(w),
        }],
        options={'ftol': 1e-11, 'maxiter': 1000},
    )
    w = result.x
    if (not result.success or not np.isfinite(w).all()
            or abs(np.sum(w) - 1.0) > 1e-7
            or np.any(w < -1e-7) or np.any(w > max_weight + 1e-7)):
        raise RuntimeError(f'Mean-variance optimization failed: {result.message}')
    return pd.Series(w, index=selected_tickers, name='weight')
