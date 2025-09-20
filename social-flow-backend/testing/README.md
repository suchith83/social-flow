# Testing (monorepo common fixtures)

This folder provides shared pytest fixtures and helpers used across services.

Quickstart
- Install test tooling (in a virtualenv):
  pip install pytest pytest-asyncio fastapi[all] requests

- Run unit tests:
  pytest -q services/recommendation-service/tests/unit

- Run integration tests for a service by pointing TEST_APP_IMPORT to the app import path:
  export TEST_APP_IMPORT=services.recommendation-service.src.main:app
  pytest -q -m integration

Fixtures
- app_client (function-scoped): creates a TestClient for the FastAPI app referenced by TEST_APP_IMPORT env var (default targets the recommendation service).
- temp_env (context manager / fixture): temporarily set environment variables during tests.
- tmp_sqlite_db (fixture): provides an isolated SQLite path and sets SF_DB_PATH env var for services using SQLite.

If you need a service-specific fixture, create it in that service's tests and reuse these helpers.
