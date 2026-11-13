"""Nine point-in-time price factors and forward-return labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .config import FEATURE_COLUMNS, HORIZON
from .data import validate_prices


def build_panel(prices: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Index output by (date, ticker); labels use *future* prices for training only.

    Each feature at t uses information <=t. A label observed at t is NOT
    available for model fitting until the close at t + horizon.
    """
    prices = validate_prices(prices)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    by_asset = []
    for ticker in prices.columns:
        price = prices[ticker]
        r = price.pct_change()
        peak = price.rolling(63, min_periods=63).max()
        columns = {
            "return_5": price.pct_change(5),
            "return_21": price.pct_change(21),
            "return_63": price.pct_change(63),
            "return_126": price.pct_change(126),
            "ma_gap_20": price / price.rolling(20).mean() - 1,
            "ma_gap_60": price / price.rolling(60).mean() - 1,
            "volatility_21": r.rolling(21).std() * np.sqrt(252),
            "volatility_63": r.rolling(63).std() * np.sqrt(252),
            "drawdown_63": price / peak - 1,
            "target_forward_return": price.shift(-horizon) / price - 1,
        }
        frame = pd.DataFrame(columns, index=prices.index)
        frame["ticker"] = ticker
        frame.index.name = "date"
        by_asset.append(frame.reset_index())
    panel = pd.concat(by_asset, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    assert tuple(panel.columns[:len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS
    return panel
