import numpy as np
import pandas as pd

from quant_research.config import FEATURE_COLUMNS


def calculate_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi / 100


def calculate_rolling_drawdown(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    rolling_high = prices.rolling(window).max()
    drawdown = prices / rolling_high - 1
    return drawdown


def stack_feature(feature_frame: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    stacked = (
        feature_frame
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name=feature_name)
        .set_index(["date", "ticker"])
    )

    return stacked


def create_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change()

    feature_frames = {
        "momentum_21": prices.pct_change(21),
        "momentum_63": prices.pct_change(63),
        "momentum_126": prices.pct_change(126),
        "momentum_252": prices.pct_change(252),
        "volatility_21": returns.rolling(21).std() * np.sqrt(252),
        "volatility_63": returns.rolling(63).std() * np.sqrt(252),
        "drawdown_63": calculate_rolling_drawdown(prices, 63),
        "ma_spread_50_200": prices.rolling(50).mean() / prices.rolling(200).mean() - 1,
        "rsi_14": calculate_rsi(prices, 14),
    }

    stacked_features = [
        stack_feature(frame, name)
        for name, frame in feature_frames.items()
    ]

    features = pd.concat(stacked_features, axis=1)
    features = features[FEATURE_COLUMNS]

    return features