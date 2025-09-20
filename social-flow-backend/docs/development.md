# Development Workflow

This page describes the minimal steps to contribute and run the project locally.

Prerequisites
- Python 3.11+, Node.js (for Node services), Docker (for infra), sqlite3 (optional)

Setup
1. Create virtualenv at repo root:
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

2. Install core tooling (optional aggregated requirements):
   python automation/setup_local.py --create-venv --apply

3. Install common libraries editable for local development:
   pip install -e common/libraries/python

Run services
- App (auth + API): python -m uvicorn app.main:app --reload --port 8000
- Recommendation: python -m uvicorn services.recommendation-service.src.main:app --reload --port 8003
- Analytics: python -m uvicorn analytics.src.main:app --reload --port 8010

Testing
- Run unit tests for a service:
  pytest services/recommendation-service/tests/unit -q
- Run full test suite (may require infra): python -m automation.ci_utils run-tests --path .

Linting & formatting
- flake8 for linting, black for formatting:
  black .
  flake8 .

Contributing
- Follow branch naming and PR conventions in CONTRIBUTING.md (if present).
- Run tests locally before opening PRs.
- Keep changes small and document breaking API changes in api-specs/.
