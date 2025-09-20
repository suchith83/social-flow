# Shared pytest fixtures for monorepo tests.

import os
import importlib
from contextlib import contextmanager
from typing import Generator, Any
import tempfile
import shutil

import pytest

from fastapi.testclient import TestClient

from testing.utils import import_app_from_string  # relative import via package name


@pytest.fixture
def temp_env(monkeypatch):
    """
    Fixture that yields a helper to set temporary environment variables.
    Usage:
        def test_x(temp_env):
            with temp_env(MY_VAR='1'):
                ...
    """
    @contextmanager
    def _set(env: dict):
        original = {}
        try:
            for k, v in env.items():
                original[k] = os.environ.get(k)
                if v is None:
                    monkeypatch.delenv(k, raising=False)
                else:
                    monkeypatch.setenv(k, str(v))
            yield
        finally:
            for k, v in original.items():
                if v is None:
                    monkeypatch.delenv(k, raising=False)
                else:
                    monkeypatch.setenv(k, v)
    return _set


@pytest.fixture
def tmp_sqlite_db(tmp_path, monkeypatch) -> str:
    """
    Provide a temporary sqlite DB path and set SF_DB_PATH environment variable.
    Returns the path as string.
    """
    db_path = tmp_path / "test.db"
    # ensure parent exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SF_DB_PATH", str(db_path))
    return str(db_path)


@pytest.fixture
def app_client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    """
    Create a TestClient for a FastAPI app specified by TEST_APP_IMPORT env var.
    Default import target: services.recommendation-service.src.main:app
    """
    import_str = os.environ.get("TEST_APP_IMPORT", "services.recommendation-service.src.main:app")
    app = import_app_from_string(import_str)
    # Ensure working dir allows service to find local data if needed
    cwd = os.getcwd()
    try:
        client = TestClient(app)
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass
