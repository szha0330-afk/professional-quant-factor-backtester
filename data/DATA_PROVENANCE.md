# Preserved historical input

`prices.csv` was retained **byte-for-byte** from the supplied older project. It contains ten named ETF price columns and dates from 2010-01-04 to 2023-12-29.

The source provider, retrieval date, dividend/split adjustments, currency, ETF inception handling, and holiday calendar were **not** supplied with this ZIP and have **not** been independently verified. Treat it as a legacy input, not an audited research dataset.

The current canonical pipeline reads this file via `python run_demo.py --source csv --csv data/prices.csv --output results_historical`, but doing so does not validate its provenance.

Before citing historical results in a CV or public project:

1. Identify the original data source and download date, then verify it allows republication.
2. Verify that columns are consistently adjusted-close prices for all ETFs.
3. Confirm dates, missing rows, asset inception dates, and any forward-filled values.
4. Recompute results from the canonical entrypoint and record hyperparameters.
5. Check execution assumptions and sensitivities to fees, spreads, and slippage.

The older `legacy/results/` files are preserved for comparison only; they are not validated outputs from the current mean-variance implementation.
