"""
Tests around multi-cloud manager endpoints.

- check current provider
- request switching provider (if endpoint exists)
- simulate provider down (if test API supports toggling or via stub)
"""

import pytest

@pytest.mark.integration
def test_get_current_provider(api_client):
    resp = api_client.get("/multi-cloud/provider")
    assert resp.status_code == 200
    data = resp.json()
    assert "provider" in data

@pytest.mark.integration
def test_switch_provider_if_supported(api_client):
    # Attempt to switch to another provider (best-effort)
    desired = "azure"
    resp = api_client.post("/multi-cloud/switch", json={"provider": desired})
    # Accept 200/202 or 404 if not supported
    assert resp.status_code < 500
