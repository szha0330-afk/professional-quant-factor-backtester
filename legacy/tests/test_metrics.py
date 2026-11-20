import pandas as pd

from quant_research.metrics import calculate_drawdown, calculate_performance_metrics


def test_calculate_drawdown():
    equity = pd.Series([1.0, 1.2, 1.1, 1.5, 1.3])
    drawdown = calculate_drawdown(equity)

    assert drawdown.iloc[0] == 0
    assert drawdown.min() < 0


def test_calculate_performance_metrics():
    index = pd.date_range("2020-01-01", periods=252, freq="B")
    returns = pd.Series([0.001] * 252, index=index)
    equity = (1 + returns).cumprod()

    metrics = calculate_performance_metrics(returns, equity)

    assert "Total Return" in metrics.index
    assert "Sharpe Ratio" in metrics.index
    assert metrics["Total Return"] > 0