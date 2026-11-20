import pandas as pd

from quant_research.portfolio import calculate_portfolio_returns


def test_calculate_portfolio_returns():
    index = pd.date_range("2020-01-01", periods=5, freq="B")

    prices = pd.DataFrame({
        "AAA": [100, 101, 102, 103, 104],
        "BBB": [100, 99, 98, 97, 96],
    }, index=index)

    weights = pd.DataFrame({
        "AAA": [0.5, 0.5, 0.5, 0.5, 0.5],
        "BBB": [0.5, 0.5, 0.5, 0.5, 0.5],
    }, index=index)

    result = calculate_portfolio_returns(prices, weights)

    assert "portfolio_return" in result.columns
    assert "equity_curve" in result.columns
    assert result["equity_curve"].iloc[-1] > 0