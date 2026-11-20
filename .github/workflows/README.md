# GitHub Actions

This directory contains the automated research testing workflow.

The workflow runs on pushes and pull requests and verifies:

- Python unit tests using pytest.
- Analytical gradient and Hessian calculations.
- Finite-difference derivative validation.

The workflow configuration is defined in tests.yml.
