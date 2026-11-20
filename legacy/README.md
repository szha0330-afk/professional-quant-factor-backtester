# Archived project artifacts

These files are kept intact as a recoverable archive of the older project, but are **not** used by the current `run_demo.py` pipeline.

- `pipeline/`: the older scripts that imported functions/configuration names absent from the revised modules.
- `tests/`: the older test files targeting those historical APIs.
- `results/`: the older cached forecasts, portfolio outcomes, reports, and plots.
- `results_original_github/`: separately retained outputs from the GitHub repository before the unified research update; these may differ from `results/` and should not be overwritten or combined.
- `CHECKSUMS.original.sha256`: the manifest from the supplied ZIP; it refers to an earlier packaging state and is **not** a valid checksum manifest for this unified folder.

Do not run the archived scripts as though they were maintained entrypoints. Do not combine the older performance reports with the new mean-variance implementation. The input `data/prices.csv` remains accessible from the repository root so its bytes can be independently audited. Use `python run_demo.py` for the maintained pipeline.
