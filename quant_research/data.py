import pandas as pd
import yfinance as yf

from quant_research.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    DATA_DIR,
    PRICE_CACHE_FILE,
)


def download_price_data(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    raw_data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw_data.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        prices = raw_data["Close"]
    else:
        prices = raw_data[["Close"]]
        prices.columns = tickers

    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna()

    return prices


def load_price_data(use_cache: bool = True) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)

    if use_cache and PRICE_CACHE_FILE.exists():
        prices = pd.read_csv(PRICE_CACHE_FILE, index_col=0, parse_dates=True)
        return prices

    prices = download_price_data()
    prices.to_csv(PRICE_CACHE_FILE)

    return prices


if __name__ == "__main__":
    data = load_price_data(use_cache=False)
    print(data.head())