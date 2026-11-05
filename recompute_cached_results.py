"""Rebuild weights, costs, charts, and metrics from stored OOS forecasts.

No model fitting or network calls; requires original results/predictions.csv
and results/feature_importance.csv plus data/prices.csv.
"""
import pandas as pd

from quant_research.config import PRICE_CACHE_FILE, RESULTS_DIR
from quant_research.backtester import evaluate_predictions
from quant_research.run_backtest import save_outputs


def main() -> None:
    prices = pd.read_csv(PRICE_CACHE_FILE, index_col=0, parse_dates=True)
    predictions = pd.read_csv(RESULTS_DIR / "predictions.csv", parse_dates=["date"])
    importance = pd.read_csv(RESULTS_DIR / "feature_importance.csv", parse_dates=["date"])

    # No holding period is observable for a trade at the dataset's last close.
    predictions = predictions.loc[predictions["date"] < prices.index.max()].copy()
    importance = importance.loc[importance["date"] < prices.index.max()].copy()
    outputs = evaluate_predictions(prices, predictions, importance)
    save_outputs(outputs)
    print("Recalculated from existing cached OOS predictions (no model retraining).")


if __name__ == "__main__":
    main()
