# Archived project artifacts

These files are kept intact as a recoverable archive of the older project, but are **not** used by the current `run_demo.py` pipeline.

- `pipeline/`: the older scripts that imported functions/configuration names absent from the revised modules.
- `tests/`: the older test files targeting those historical APIs.
- `results/`: the older cached forecasts, portfolio outcomes, reports, and plots.
- `CHECKSUMS.original.sha256`: the manifest from the supplied ZIP; it refers to an earlier packaging state and is **not** a valid checksum manifest for this unified folder.

Do not run the archived scripts as though they were maintained entrypoints. Do not combine the older performance reports with the new mean-variance implementation. The input `data/prices.csv` remains accessible at its original relative path so its bytes can be independently audited.
