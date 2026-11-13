"""Expanding-window Random Forest prediction with strict horizon embargo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import FEATURE_COLUMNS, HORIZON
from .factors import build_panel
from .data import validate_prices


def rebalance_dates(dates: pd.DatetimeIndex, first_index: int) -> list[pd.Timestamp]:
    """Take last available trading day in each month, from a minimum history."""
    if first_index < 0 or first_index >= len(dates):
        return []
    filtered = dates[first_index:]
    return [pd.Timestamp(x) for x in pd.Series(filtered, index=filtered).groupby(filtered.to_period("M")).last()]


def generate_predictions(
    prices: pd.DataFrame,
    horizon: int = HORIZON,
    min_history: int = 260,
    n_estimators: int = 25,
    min_samples_leaf: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Forecast forward horizon returns at month-end signal dates.

    For a signal at date index k, training examples are restricted to dates
    at or before index k-horizon; their *full labels* are known at signal k.
    Test samples at k never enter the training set even through their labels.
    """
    prices = validate_prices(prices)
    if horizon < 1 or min_history < horizon + 130 or n_estimators < 1:
        raise ValueError("Invalid training parameters")
    dates = prices.index
    if len(dates) <= min_history + 2:
        raise ValueError("Not enough dates to obtain out-of-sample signals")
    panel = build_panel(prices, horizon=horizon)
    Xcols = list(FEATURE_COLUMNS)
    predictions: list[dict] = []
    for signal_date in rebalance_dates(dates, min_history):
        k = int(dates.get_loc(signal_date))
        cutoff_date = dates[k - horizon]
        train = panel.loc[panel.index.get_level_values("date") <= cutoff_date]
        train = train.dropna(subset=Xcols + ["target_forward_return"])
        test = panel.xs(signal_date, level="date").dropna(subset=Xcols)
        if len(train) < max(100, len(prices.columns) * 30) or len(test) < 2:
            continue
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(train[Xcols], train["target_forward_return"])
        yhat = model.predict(test[Xcols])
        for ticker, forecast in zip(test.index, yhat):
            if np.isfinite(forecast):
                predictions.append({"date": signal_date, "ticker": ticker,
                                    "predicted_forward_return": float(forecast),
                                    "training_cutoff": cutoff_date})
    if not predictions:
        raise ValueError("No valid signals: increase data length or reduce min_history")
    return pd.DataFrame(predictions).sort_values(["date", "ticker"]).reset_index(drop=True)
