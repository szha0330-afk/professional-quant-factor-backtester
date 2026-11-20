"""Smoke and provenance tests for the supported, unified workflow."""

import json
import numpy as np
import pandas as pd

from quant_research.config import TICKERS
from quant_research.data import read_prices_csv, synthetic_prices
from run_demo import run


def test_preserved_original_universe_and_prices_are_consistent():
    original = read_prices_csv("data/prices.csv")
    assert tuple(original.columns) == TICKERS
    assert original.index[0] == pd.Timestamp("2010-01-04")
    assert original.index[-1] == pd.Timestamp("2023-12-29")
    assert len(original) > 3000


def test_canonical_end_to_end_demo(tmp_path):
    prices = synthetic_prices(periods=345, tickers=("SPY", "QQQ", "TLT", "GLD"), seed=8)
    summary = run(prices, tmp_path, estimators=5, top_n=3, cost_bps=10.0, synthetic=True)
    assert set(summary.index) == {"inverse_vol", "mean_variance"}
    for method in summary.index:
        weights = pd.read_csv(tmp_path / f"signal_weights_{method}.csv")
        allocations = weights.drop(columns="signal_date")
        assert np.allclose(allocations.sum(axis=1), 1, atol=1e-7)
        assert (allocations.to_numpy() <= 0.5 + 1e-7).all()
    errors = json.loads((tmp_path / "derivative_check.json").read_text())
    assert errors["max_gradient_abs_error"] < 1e-8
    assert errors["max_hessian_abs_error"] < 1e-8
    assert "SYNTHETIC" in (tmp_path / "DATA_PROVENANCE.txt").read_text()
