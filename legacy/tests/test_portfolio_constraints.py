"""Economic constraints and transaction timing for the portfolio engine."""
import numpy as np
import pandas as pd
import pytest

from quant_research.portfolio import (
    enforce_weight_cap,
    calculate_inverse_vol_weights,
    calculate_portfolio_returns,
    build_portfolio_weights,
)


def test_cap_redistributes_without_rebreaking_limit():
    result = enforce_weight_cap(pd.Series([0.8, 0.1, 0.1], index=list("ABC")), 0.50)
    assert result.sum() == pytest.approx(1.0)
    assert result.max() <= 0.5 + 1e-10
    assert result["A"] == pytest.approx(0.50)
    assert result["B"] == pytest.approx(0.25)
    assert result["C"] == pytest.approx(0.25)


def test_infeasible_position_cap_fails_explicitly():
    with pytest.raises(ValueError, match="infeasible"):
        enforce_weight_cap(pd.Series([1.0]), 0.5)


def test_inverse_vol_weights_obey_cap():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = pd.DataFrame({
        "A": 100 * 1.002 ** np.arange(100),
        "B": 100 * (1 + .002 * np.sin(np.arange(100))),
        "C": 100 * (1 + .01 * np.sin(np.arange(100))),
    }, index=dates)
    weights = calculate_inverse_vol_weights(prices, dates[-1], ["A", "B", "C"])
    assert weights.sum() == pytest.approx(1.0)
    assert (weights <= 0.5 + 1e-10).all()
    assert (weights >= 0).all()


def test_drift_is_not_daily_rebalanced_or_charged():
    days = pd.date_range("2020-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100, 120, 120], "B": [100, 100, 100]}, index=days)
    targets = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=days)
    result = calculate_portfolio_returns(
        prices, targets, transaction_cost=0.001,
        rebalance_dates=pd.DatetimeIndex([days[0], days[2]])
    )
    assert result.loc[days[0], "gross_return"] == pytest.approx(0.0)
    assert result.loc[days[0], "turnover"] == pytest.approx(1.0)
    assert result.loc[days[1], "gross_return"] == pytest.approx(0.1)
    assert result.loc[days[1], "turnover"] == pytest.approx(0.0)
    # After A rises, pre-rebalance A weight = 0.6/1.1, so both sides trade.
    expected_turnover = 2 * (0.6/1.1 - 0.5)
    assert result.loc[days[2], "turnover"] == pytest.approx(expected_turnover)
    assert result.loc[days[2], "transaction_cost"] == pytest.approx(expected_turnover * .001)


def test_rebalance_liquidates_assets_no_longer_selected():
    days = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"A": [100, 101, 102, 103], "B": [100, 99, 98, 97],
                           "C": [100, 100, 100, 100]}, index=days)
    predictions = pd.DataFrame([
        {"date": days[0], "ticker": "A", "predicted_forward_return": 2},
        {"date": days[0], "ticker": "B", "predicted_forward_return": 1},
        {"date": days[2], "ticker": "B", "predicted_forward_return": 2},
        {"date": days[2], "ticker": "C", "predicted_forward_return": 1},
    ])
    # One asset at 50% is infeasible; two assets at 50% is valid.
    targets = build_portfolio_weights(prices, predictions, top_n=2)
    assert targets.loc[days[2], "A"] == pytest.approx(0.0)
    assert targets.loc[days[2], "B"] == pytest.approx(0.5)
    assert targets.loc[days[2], "C"] == pytest.approx(0.5)
    assert targets.loc[days[3], "A"] == pytest.approx(0.0)
