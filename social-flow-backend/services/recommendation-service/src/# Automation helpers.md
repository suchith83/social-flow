# Automation helpers

This folder contains utility scripts and a development docker-compose for the Social Flow monorepo.

Files
- setup_local.py — create a virtualenv, collect runtime requirement files across services, optionally install them and provide commands to run core services.
- ci_utils.py — lightweight helpers to run tests, linters and generate a summary (used by CI runners).
- docker-compose.dev.yml — example docker-compose for local development: Postgres, Redis, MinIO.

Quick examples
- Inspect what would be installed (dry run):
  python automation/setup_local.py --dry-run

- Create venv and install requirements (applies changes):
  python automation/setup_local.py --create-venv --apply

- Run tests for a service:
  python -m automation.ci_utils run-tests --path services/recommendation-service

Notes
- These helpers are intentionally conservative. Review the commands they will run before using `--apply`.
- For full local end-to-end: start `docker-compose -f automation/docker-compose.dev.yml up` then start services with their dev commands.
