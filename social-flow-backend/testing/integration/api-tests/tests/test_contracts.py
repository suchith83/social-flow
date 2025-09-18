"""
Contract tests: strictly validate responses against Pydantic models.

This module uses the Pydantic models defined in schemas/models.py
to validate important API contracts in a strict manner.
"""

import pytest
from schemas.models import UserModel
import jsonschema
from jsonschema import validate

@pytest.mark.contract
def test_user_model_in_user_info(api_client):
    resp = api_client.get("/auth/me")
    if resp.status_code == 401:
        pytest.skip("auth/me requires auth; skip when unauthenticated")
    assert resp.status_code == 200
    user = resp.json()
    parsed = UserModel.parse_obj(user)
    assert parsed.username
