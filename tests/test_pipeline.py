import numpy as np
import pandas as pd
import pytest

from quant_research.config import FEATURE_COLUMNS
from quant_research.data import synthetic_prices, validate_prices
from quant_research.factors import build_panel
from quant_research.model import generate_predictions, rebalance_dates
from quant_research.portfolio import capped_simplex, build_portfolio_weights
from quant_research.backtest import simulate_portfolio
from quant_research.metrics import max_drawdown, summarize


def small_prices():
    return synthetic_prices(periods=340, tickers=("SPY", "QQQ", "TLT", "GLD"), seed=15)


def test_features_are_past_only_and_labels_forward():
    p = small_prices()
    panel = build_panel(p, horizon=21)
    assert len(FEATURE_COLUMNS) == 9
    when = p.index[180]
    later = p.copy()
    later.loc[p.index[181]:, "SPY"] *= 5
    changed = build_panel(later)
    pd.testing.assert_series_equal(panel.loc[(when, "SPY"), list(FEATURE_COLUMNS)],
                                   changed.loc[(when, "SPY"), list(FEATURE_COLUMNS)])
    expected = p.loc[p.index[201], "SPY"] / p.loc[when, "SPY"] - 1
    assert np.isclose(panel.loc[(when, "SPY"), "target_forward_return"], expected)


def test_forecast_horizon_embargo_and_reproducible():
    p = small_prices()
    predictions = generate_predictions(p, n_estimators=5)
    assert not predictions.empty
    for _, row in predictions.iterrows():
        i = p.index.get_loc(row.date)
        assert row.training_cutoff == p.index[i - 21]
    same = generate_predictions(p, n_estimators=5)
    pd.testing.assert_frame_equal(predictions, same)


def test_capped_projection_and_construction():
    w = capped_simplex([0.8, 0.15, 0.05], 0.5)
    assert np.isclose(w.sum(), 1)
    assert max(w) <= 0.5 + 1e-8
    p = small_prices()
    predictions = generate_predictions(p, n_estimators=5)
    for method in ("inverse_vol", "mean_variance"):
        targets = build_portfolio_weights(p, predictions, top_n=3,
                                          allocation_method=method)
        assert np.allclose(targets.sum(axis=1), 1)
        assert np.max(targets.to_numpy()) <= 0.5 + 1e-8
        assert np.min(targets.to_numpy()) >= -1e-8


def test_signal_is_only_executed_next_day_and_cost_deducted():
    dates = pd.bdate_range("2020-01-01", periods=6)
    p = pd.DataFrame({"SPY": [100, 100, 110, 110, 110, 110],
                      "TLT": [100, 100, 100, 100, 100, 100]}, index=dates)
    w = pd.DataFrame({"SPY": [0.5], "TLT": [0.5]}, index=[dates[1]])
    result = simulate_portfolio(p, w, transaction_cost_bps=10)
    assert result.daily.loc[dates[1], "portfolio_return"] == 0.0
    assert np.isclose(result.daily.loc[dates[2], "gross_return"], 0.05)
    assert np.isclose(result.daily.loc[dates[2], "cost_fraction"], 0.001)
    assert np.isclose(result.daily.loc[dates[2], "portfolio_return"], 0.999 * 1.05 - 1)
    assert result.trades.iloc[0]["execution_date"] == dates[2]


def test_backtest_turnover_uses_drifted_weights():
    dates = pd.bdate_range("2020-01-01", periods=7)
    p = pd.DataFrame({"SPY": [100, 100, 110, 110, 110, 110, 110],
                      "TLT": [100, 100, 100, 100, 100, 100, 100]}, index=dates)
    signals = pd.DataFrame({"SPY": [0.5, 0.5], "TLT": [0.5, 0.5]},
                           index=[dates[1], dates[3]])
    sim = simulate_portfolio(p, signals, transaction_cost_bps=0)
    # After SPY rises 10%, holdings drift to (0.55/1.05, 0.5/1.05)
    expected = abs(0.5 - 0.55 / 1.05) + abs(0.5 - 0.5 / 1.05)
    assert np.isclose(sim.trades.iloc[1]["turnover"], expected)


def test_basic_metrics():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert np.isclose(max_drawdown(eq), -0.25)
    daily = pd.Series([0.0, 0.2, -0.25, 1.1 / 0.9 - 1])
    summary = summarize(daily, eq)
    assert np.isclose(summary["total_return"], 0.1)


def test_price_validation_rejects_missing_and_unsorted():
    p = small_prices()
    p.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_prices(p)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_prices(small_prices().iloc[::-1])


def test_bad_cap_fails():
    with pytest.raises(ValueError, match="infeasible"):
        capped_simplex(np.ones(2), 0.4)


def test_future_changes_do_not_alter_earlier_model_forecast():
    prices = small_prices()
    p1 = generate_predictions(prices, n_estimators=5)
    first_date = p1["date"].iloc[0]
    modified = prices.copy()
    modified.loc[modified.index > first_date, "SPY"] *= 3.0
    p2 = generate_predictions(modified, n_estimators=5)
    earlier1 = p1[p1.date <= first_date].reset_index(drop=True)
    earlier2 = p2[p2.date <= first_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier1, earlier2)
