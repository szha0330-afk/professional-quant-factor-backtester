"""Command line comparison of inverse volatility and mean-variance portfolios.

Default mode is fully offline using simulated prices. Its resulting statistics
MUST NOT be reported as historical real-world strategy performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant_research.config import BENCHMARK, HORIZON, MAX_WEIGHT, TOP_N
from quant_research.data import synthetic_prices, read_prices_csv, download_adjusted_prices
from quant_research.model import generate_predictions
from quant_research.portfolio import build_portfolio_weights
from quant_research.backtest import simulate_portfolio
from quant_research.metrics import summarize
from quant_research.diagnostics import derivative_diagnostics


def run(prices: pd.DataFrame, output: Path, estimators: int, top_n: int, cost_bps: float,
        synthetic: bool) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    predictions = generate_predictions(prices, n_estimators=estimators)
    predictions.to_csv(output / "predictions.csv", index=False)
    result_rows = []
    curves = {}
    for method in ("inverse_vol", "mean_variance"):
        targets = build_portfolio_weights(prices, predictions, allocation_method=method,
                                          top_n=top_n, max_weight=MAX_WEIGHT, horizon=HORIZON)
        targets.to_csv(output / f"signal_weights_{method}.csv")
        sim = simulate_portfolio(prices, targets, transaction_cost_bps=cost_bps,
                                 benchmark=BENCHMARK if BENCHMARK in prices.columns else prices.columns[0])
        sim.daily.to_csv(output / f"daily_{method}.csv")
        sim.trades.to_csv(output / f"trades_{method}.csv", index=False)
        first = sim.trades["execution_date"].min()
        active = sim.daily.loc[first:]
        perf = summarize(active["portfolio_return"], active["equity"])
        perf.update({"method": method, "total_turnover": float(sim.trades["turnover"].sum()),
                     "rebalance_count": len(sim.trades)})
        result_rows.append(perf)
        curves[method] = sim.daily["equity"]
        curves["benchmark"] = sim.daily["benchmark_equity"]
    metrics = pd.DataFrame(result_rows).set_index("method")
    metrics.to_csv(output / "strategy_metrics.csv")
    curves_frame = pd.DataFrame(curves)
    curves_frame.to_csv(output / "equity_curves.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    curves_frame.plot(ax=ax, linewidth=1.7)
    ax.set(ylabel="Portfolio value (initial = 1)", xlabel="Date",
           title="SIMULATED example — NOT observed market returns" if synthetic
                 else "Historical research backtest — check data and assumptions")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    sample_cov = np.array([[0.04, 0.006, 0.004],
                           [0.006, 0.03, 0.008],
                           [0.004, 0.008, 0.05]], dtype=float)
    sample_w = np.array([0.3, 0.4, 0.3])
    sample_mu = np.array([0.02, 0.015, 0.025])
    report = derivative_diagnostics(sample_w, sample_mu, sample_cov)
    (output / "derivative_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "DATA_PROVENANCE.txt").write_text(
        "SYNTHETIC RANDOM PRICES. RESULTS ARE NOT REAL-MARKET RETURNS.\n" if synthetic else
        "Prices provided by user or market data vendor. Verify adjusted-close data, survivorship, and execution assumptions.\n",
        encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("synthetic", "csv", "yahoo"), default="synthetic")
    p.add_argument("--csv", type=Path, help="Wide adjusted-close CSV with Date column")
    p.add_argument("--days", type=int, default=650)
    p.add_argument("--estimators", type=int, default=20)
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()
    if args.source == "synthetic":
        prices = synthetic_prices(periods=args.days)
    elif args.source == "csv":
        if args.csv is None:
            p.error("--csv must be specified with --source csv")
        prices = read_prices_csv(args.csv)
    else:
        prices = download_adjusted_prices()
    comparison = run(prices, args.output, args.estimators, args.top_n,
                     args.cost_bps, args.source == "synthetic")
    print("SIMULATED RESULTS — NOT HISTORICAL" if args.source == "synthetic" else
          "RESEARCH RESULTS — NOT VERIFIED INVESTMENT PERFORMANCE")
    print(comparison.round(4).to_string())
    print(f"Saved files in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
