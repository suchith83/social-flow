# Data: Migrations, Seeds & Fixtures

This folder provides simple, portable migration and seed files intended for local development and CI smoke-tests.

Quick start (SQLite)
1. Create a local sqlite DB file:
   mkdir -p .data
   sqlite3 .data/dev.db < data/migrations/sqlite/001_initial_schema.sql

2. Seed development data:
   sqlite3 .data/dev.db < data/seeds/development/seed_data.sql

3. Inspect:
   sqlite3 .data/dev.db "SELECT * FROM users;"

Quick start (Postgres)
- Use psql and apply the same SQL (minor syntax differences may apply). Example:
  psql postgres://user:pass@localhost:5432/dbname -f data/migrations/sqlite/001_initial_schema.sql

Notes
- These SQL files are convenience scripts for local development and tests. For production use, convert to proper migration tooling (Alembic, Flyway) and use parametrized secrets from your infrastructure.
- Fixtures for video uploads live under data/fixtures/test-videos. Do not commit large binaries.
