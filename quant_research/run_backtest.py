from quant_research.config import RESULTS_DIR
from quant_research.data import load_price_data
from quant_research.backtester import run_walk_forward_backtest
from quant_research.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_feature_importance,
)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading price data...")
    prices = load_price_data(use_cache=True)

    print("Running walk-forward ML factor backtest...")
    outputs = run_walk_forward_backtest(prices)

    print("Saving result files...")
    outputs["performance_report"].index.name = "Metric"
    outputs["predictions"].to_csv(RESULTS_DIR / "predictions.csv", index=False)
    outputs["feature_importance"].to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    outputs["weights"].to_csv(RESULTS_DIR / "weights.csv")
    outputs["backtest_results"].to_csv(RESULTS_DIR / "backtest_results.csv")
    outputs["performance_report"].to_csv(RESULTS_DIR / "performance_report.csv")
    outputs["performance_report"].to_csv(
    RESULTS_DIR / "performance_report.csv"
)

    print("Generating plots...")
    plot_equity_curve(outputs["backtest_results"])
    plot_drawdown(outputs["backtest_results"])
    plot_feature_importance(outputs["feature_importance"])

    print("\nBacktest Completed")
    print("------------------")
    print(outputs["performance_report"])


if __name__ == "__main__":
    main()