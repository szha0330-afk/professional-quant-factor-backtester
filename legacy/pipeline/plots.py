import pandas as pd
import matplotlib.pyplot as plt

from quant_research.metrics import calculate_drawdown
from quant_research.config import BENCHMARK, RESULTS_DIR


def plot_equity_curve(results: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(results.index, results["equity_curve"], label="Random Forest Factor Portfolio")
    plt.plot(results.index, results["benchmark_equity"], label=f"{BENCHMARK} Buy and Hold")
    plt.title("ML Risk-Controlled Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.savefig(RESULTS_DIR / "equity_curve.png")
    plt.close()


def plot_drawdown(results: pd.DataFrame) -> None:
    portfolio_drawdown = calculate_drawdown(results["equity_curve"])
    benchmark_drawdown = calculate_drawdown(results["benchmark_equity"])

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_drawdown.index, portfolio_drawdown, label="Portfolio Drawdown")
    plt.plot(benchmark_drawdown.index, benchmark_drawdown, label=f"{BENCHMARK} Drawdown")
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.savefig(RESULTS_DIR / "drawdown.png")
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    average_importance = (
        feature_importance
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(average_importance.index, average_importance.values)
    plt.title("Average Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png")
    plt.close()