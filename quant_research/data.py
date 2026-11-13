"""Input price validation, reproducible synthetic prices, and optional market download."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import TICKERS


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Require a complete, strictly-positive adjusted-close matrix.

    Never silently fill missing observations: a forward-fill can manufacture
    returns on non-trading days and contaminate covariance estimates.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a DatetimeIndex")
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("At least two assets and some rows are required")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be strictly increasing and unique")
    if prices.columns.has_duplicates:
        raise ValueError("Duplicate ticker columns are not allowed")
    result = prices.astype(float).copy()
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError("All prices must be finite and strictly positive")
    return result


def synthetic_prices(
    periods: int = 900,
    tickers: tuple[str, ...] = TICKERS,
    seed: int = 2026,
    start: str = "2018-01-02",
) -> pd.DataFrame:
    """Simulated series for software checks; NOT historical or observed prices."""
    if periods < 220 or len(tickers) < 2:
        raise ValueError("Need >=220 trading days and >=2 assets")
    rng = np.random.default_rng(seed)
    n = len(tickers)
    common = rng.normal(0.00015, 0.008, size=(periods, 1))
    idio = rng.normal(0.0001, 0.008, size=(periods, n))
    betas = np.linspace(0.25, 1.15, n)
    log_return = common * betas + idio
    prices = 100.0 * np.exp(np.cumsum(log_return, axis=0))
    return validate_prices(pd.DataFrame(prices, index=pd.bdate_range(start, periods=periods), columns=tickers))


def read_prices_csv(path: str | Path) -> pd.DataFrame:
    """Read wide CSV: Date,SPY,QQQ,... with adjusted daily close values."""
    data = pd.read_csv(path, parse_dates=["Date"])
    return validate_prices(data.set_index("Date"))


def download_adjusted_prices(
    tickers: tuple[str, ...] = TICKERS,
    start: str = "2016-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Optional network path; never used in offline demos and unit tests."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("pip install yfinance to use Yahoo Finance downloads") from exc
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Market data download returned no observations")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=list(tickers)) if len(tickers) > 1 else close.rename(columns={"Close": tickers[0]})
    return validate_prices(close.dropna())
