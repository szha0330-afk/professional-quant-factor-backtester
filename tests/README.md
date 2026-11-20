# Automated tests

Run the deterministic tests from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The test suite checks the unified entrypoint, forecast/backtest pipeline, and constrained portfolio optimization. The separate derivative audit validates analytic gradient and Hessian calculations:

```bash
python -m examples.inspect_derivatives
```

The GitHub Actions workflow in `.github/workflows/tests.yml` runs both checks for pushes and pull requests.
