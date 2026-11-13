import numpy as np
import pandas as pd
import pytest

from quant_research.optimization import (
    objective, gradient, hessian, variance_contributions, optimize_mean_variance,
)
from quant_research.diagnostics import (
    derivative_diagnostics, euler_volatility_contributions,
)


def test_analytic_gradient_matches_finite_difference():
    mu = np.array([0.02, 0.04, 0.03])
    sigma = np.array([[0.05, 0.01, 0], [0.01, 0.07, 0.02], [0, 0.02, 0.04]])
    w = np.array([0.2, 0.4, 0.4])
    eps = 1e-6
    finite_diff = np.array([(objective(w + eps * np.eye(3)[i], mu, sigma, 8.0) -
                             objective(w - eps * np.eye(3)[i], mu, sigma, 8.0)) / (2 * eps)
                            for i in range(3)])
    np.testing.assert_allclose(gradient(w, mu, sigma, 8.0), finite_diff, atol=1e-8)


def test_analytic_hessian_matches_numeric_and_is_convex():
    sigma = np.array([[0.05, 0.01], [0.01, 0.07]])
    w = np.array([0.4, 0.6])
    mu = np.array([0.01, 0.02])
    np.testing.assert_allclose(hessian(sigma, 8.0), 8 * sigma)
    report = derivative_diagnostics(w, mu, sigma)
    assert report["max_gradient_abs_error"] < 1e-8
    assert report["max_hessian_abs_error"] < 1e-8
    assert report["hessian_min_eigenvalue"] > 0
    np.testing.assert_allclose(variance_contributions(w, sigma).sum(), w @ sigma @ w)
    np.testing.assert_allclose(euler_volatility_contributions(w, sigma).sum(),
                               np.sqrt(w @ sigma @ w))


def test_optimization_constraints_and_no_future_price_dependence():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=120)
    changes = rng.normal(0.0002, 0.009, size=(120, 3))
    prices = pd.DataFrame(100 * np.cumprod(1 + changes, axis=0),
                          index=dates, columns=["AAA", "BBB", "CCC"])
    signal_date = dates[90]
    predicted = pd.Series({"AAA": 0.01, "BBB": 0.02, "CCC": 0.015})
    args = dict(date=signal_date, selected_tickers=["AAA", "BBB", "CCC"],
                predicted_returns=predicted, max_weight=0.5)
    w1 = optimize_mean_variance(prices, **args)
    assert abs(w1.sum() - 1) < 1e-7
    assert w1.between(-1e-7, 0.5 + 1e-7).all()
    changed = prices.copy()
    changed.loc[dates[91]:, "AAA"] *= 100
    w2 = optimize_mean_variance(changed, **args)
    np.testing.assert_allclose(w1, w2, atol=1e-10)


def test_infeasible_weight_cap_raises():
    p = pd.DataFrame({"A": range(100, 200), "B": range(110, 210)},
                     index=pd.bdate_range("2022-01-01", periods=100))
    with pytest.raises(ValueError, match="infeasible"):
        optimize_mean_variance(p, p.index[-1], ["A", "B"],
                               pd.Series({"A": 0.02, "B": 0.02}), max_weight=0.4)
