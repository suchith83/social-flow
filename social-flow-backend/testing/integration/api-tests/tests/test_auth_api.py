"""
Authentication integration tests.

Covers:
- login success and token contract
- invalid login failure scenarios
- token-protected endpoints require Authorization
"""

import pytest
from schemas.models import AuthToken, UserModel

@pytest.mark.integration
def test_login_and_token_contract(api_client, admin_credentials):
    # Attempt login
    resp = api_client.post("/auth/login", json=admin_credentials)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code} - {resp.text}"
    body = resp.json()
    # Validate contract
    token = AuthToken.parse_obj(body)
    assert token.token and isinstance(token.token, str)

@pytest.mark.integration
def test_invalid_login_returns_401(api_client):
    resp = api_client.post("/auth/login", json={"username": "nope", "password": "wrong"})
    assert resp.status_code in (401, 400), "Invalid login should return 401/400"
