"""
Pytest configuration and fixtures for API integration tests.

- Provides a robust API client as a session-scoped fixture
- Seeds test users / test data (best-effort) before tests depending on available endpoints
- Provides cleanup fixtures and helper utilities
"""

import os
import json
import pytest
from pathlib import Path
from dotenv import load_dotenv

# load .env if present
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

from clients.api_client import ApiClient
from utils.test_helpers import random_string

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api")

@pytest.fixture(scope="session")
def api_base_url():
    return API_BASE.rstrip("/")

@pytest.fixture(scope="session")
def api_client(api_base_url):
    """
    Provide a session-scoped ApiClient that reuses a requests.Session.
    Tests should use this client for stable connection reuse and built-in retries.
    """
    client = ApiClient(base_url=api_base_url)
    yield client
    client.close()

@pytest.fixture(scope="session")
def admin_credentials():
    """Return admin credentials from env or defaults for seeding."""
    return {
        "username": os.getenv("TEST_ADMIN_USERNAME", "admin"),
        "password": os.getenv("TEST_ADMIN_PASSWORD", "password123"),
    }

@pytest.fixture(scope="session")
def seeded_user(api_client, admin_credentials):
    """
    Create a disposable test user via API (best-effort).
    If create-user endpoint not available, returns a generated username/password without creating.
    """
    username = f"test_{random_string(8)}"
    password = "P@ssw0rd!"  # strong test password

    # try to create via admin endpoint if available
    created = False
    try:
        # try to login as admin first (if auth required)
        # endpoint names are examples; adjust to your API contract
        login_resp = api_client.post("/auth/login", json=admin_credentials)
        if login_resp.status_code == 200 and "token" in login_resp.json():
            token = login_resp.json()["token"]
            api_client.set_bearer_token(token)
        resp = api_client.post("/testing/create-user", json={"username": username, "password": password})
        if resp.status_code in (200, 201):
            created = True
    except Exception:
        # best-effort: ignore if not available
        created = False

    yield {"username": username, "password": password, "created": created}

    # Teardown: attempt to remove user if created
    if created:
        try:
            api_client.post("/testing/delete-user", json={"username": username})
        except Exception:
            pass
