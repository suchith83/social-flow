# Unit tests for endpoints
"""
Unit tests for Mobile Optimized API.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from mobile_backend.apis.mobile_optimized.routes import router
from mobile_backend.apis.mobile_optimized.database import get_db

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_list_items_default():
    response = client.get("/mobile-optimized/items")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_delta_sync():
    payload = {"last_sync_timestamp": "2025-01-01T00:00:00Z"}
    response = client.post("/mobile-optimized/delta-sync", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "updated_items" in data
    assert "deleted_item_ids" in data
    assert "server_timestamp" in data
