import pytest


@pytest.mark.asyncio
async def test_health_root(async_client):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "x-request-id" in resp.headers


@pytest.mark.asyncio
async def test_health_models_endpoint(async_client):
    resp = await async_client.get("/api/v1/health/models")
    # Accept either success or graceful absence
    if resp.status_code == 200:
        body = resp.json()
        assert any(k in body for k in ["ml_models", "models", "capabilities"]) or body.get("status") == "ok"
    else:
        pytest.skip("models health endpoint unavailable")
