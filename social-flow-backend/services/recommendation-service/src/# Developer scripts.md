# Developer scripts

This folder contains small helpers for local development.

Available scripts:
- setup_dev_env.py  — create virtualenv, aggregate runtime requirements and optionally install.
- run_local.sh      — start local docker-compose dev stack (deployment/docker-compose.dev.yml).
- migrate_and_seed.py — apply SQLite migrations and development seeds to .data/dev.db.
- generate_self_signed.sh — generate self-signed TLS cert/key for local dev.
- cleanup.py        — stop compose stack and remove temporary files.

Examples:
- python scripts/setup_dev_env.py --create-venv --apply
- bash scripts/run_local.sh
- python scripts/migrate_and_seed.py
- bash scripts/generate_self_signed.sh

Note: Review scripts before running. They are designed to be safe and idempotent for development.
