# Quality Assurance

This folder contains configuration and lightweight tooling to enforce code quality across the monorepo.

Quickstart (local)
1. Install developer tools (recommended):
   python -m pip install -U black isort flake8 pytest coverage pre-commit

2. Install pre-commit hooks:
   pre-commit install --config quality-assurance/pre-commit-config.yaml

3. Run checks:
   python quality-assurance/tools/run_quality_checks.py

CI
- An example GitHub Actions job is provided at quality-assurance/ci/quality_checks.yml.
- You can move that file to `.github/workflows/quality_checks.yml` to enable it.

Notes
- Configs aim to be conservative so they work across many services in this monorepo.
- Adjust excludes and per-service overrides as needed.
