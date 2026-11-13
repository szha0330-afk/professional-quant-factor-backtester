"""Portfolio construction: inverse-volatility baseline and constrained MVO."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_WEIGHT, TOP_N, VOL_LOOKBACK
from .optimization import optimize_mean_variance


def capped_simplex(raw: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {sum(w)=1, 0<=w<=cap} via bisection."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or not 0 < cap <= 1:
        raise ValueError("raw must be finite vector and cap in (0,1]")
    if raw.size * cap < 1 - 1e-10:
        raise ValueError("Cap is infeasible for the number of assets")
    lo, hi = float(raw.min() - cap - 1), float(raw.max() + 1)
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(raw - mid, 0, cap).sum() > 1:
            lo = mid
        else:
            hi = mid
    weights = np.clip(raw - (lo + hi) / 2, 0, cap)
    if abs(weights.sum() - 1) > 1e-8:
        raise RuntimeError("Projection failed numerical feasibility check")
    return weights


def inverse_volatility_weights(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    tickers: list[str],
    lookback: int = VOL_LOOKBACK,
    max_weight: float = MAX_WEIGHT,
) -> pd.Series:
    if not tickers:
        raise ValueError("At least one ticker is required")
    if date not in prices.index:
        raise ValueError("date must be an observed trading day")
    observations = prices.loc[:date, tickers].tail(lookback + 1).pct_change(fill_method=None).dropna()
    if len(observations) < 30:
        raise ValueError("Insufficient history for inverse volatility")
    vols = observations.std(ddof=1).to_numpy(dtype=float)
    if not np.isfinite(vols).all():
        raise ValueError("Invalid volatility")
    raw = 1.0 / np.maximum(vols, 1e-8)
    raw /= raw.sum()
    return pd.Series(capped_simplex(raw, max_weight), index=tickers, name="weight")


def build_portfolio_weights(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    top_n: int = TOP_N,
    allocation_method: str = "inverse_vol",
    max_weight: float = MAX_WEIGHT,
    lookback: int = VOL_LOOKBACK,
    horizon: int = 21,
    risk_aversion: float = 8.0,
) -> pd.DataFrame:
    """Daily index; nonzero weights only on rebalance signal dates.

    These are *orders submitted after signal-date close*. Execution and holding
    logic is handled independently by the next-session backtest simulator.
    """
    if top_n < 1 or top_n * max_weight < 1 - 1e-10:
        raise ValueError("top_n too small for the position cap")
    if allocation_method not in ("inverse_vol", "mean_variance"):
        raise ValueError(f"Unknown allocation method: {allocation_method}")
    required = {"date", "ticker", "predicted_forward_return"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction columns missing: {required - set(predictions.columns)}")
    if predictions.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker predictions")
    columns = list(prices.columns)
    targets = pd.DataFrame(0.0, index=prices.index, columns=columns)
    signal_dates = []
    for date, group in predictions.groupby("date", sort=True):
        date = pd.Timestamp(date)
        if date not in prices.index:
            raise ValueError(f"Signal date {date} is absent from prices")
        group = group[group["ticker"].isin(columns)].copy()
        group = group[np.isfinite(group["predicted_forward_return"].astype(float))]
        selected = group.sort_values(["predicted_forward_return", "ticker"],
                                     ascending=[False, True]).head(top_n)
        tickers = selected["ticker"].tolist()
        if len(tickers) * max_weight < 1 - 1e-10:
            continue
        if allocation_method == "inverse_vol":
            w = inverse_volatility_weights(prices, date, tickers, lookback, max_weight)
        else:
            w = optimize_mean_variance(
                prices=prices, date=date, selected_tickers=tickers,
                predicted_returns=selected.set_index("ticker")["predicted_forward_return"],
                lookback=lookback, horizon=horizon, max_weight=max_weight,
                risk_aversion=risk_aversion,
            )
        targets.loc[date, w.index] = w.to_numpy()
        signal_dates.append(date)
    if not signal_dates:
        raise ValueError("Could not construct any feasible portfolio")
    result = targets.loc[signal_dates]
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Portfolio weights are not fully invested")
    if (result > max_weight + 1e-8).any().any():
        raise RuntimeError("Position cap violated")
    result.index.name = "signal_date"
    return result
