# Quantitative research package

`quant_research` contains the maintained, modular implementation for ETF factor research and backtesting.

- `data.py`, `config.py`, and `factors.py`: input validation, settings, nine price factors, and forward-return labels.
- `model.py`: expanding-window Random Forest forecasts of 21-trading-day returns.
- `optimization.py` and `portfolio.py`: constrained mean-variance allocation, analytic derivatives, and inverse-volatility weights.
- `backtest.py` and `metrics.py`: walk-forward portfolio evaluation, costs, and performance summaries.
- `diagnostics.py`: finite-difference derivative checks and risk diagnostics.

The supported command-line entrypoint is `python run_demo.py` from the repository root. Historical scripts are preserved under `legacy/` and are not part of this maintained package.
