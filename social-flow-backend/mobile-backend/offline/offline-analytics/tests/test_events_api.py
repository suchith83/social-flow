# Unit tests for analytics API routes
"""
Basic integration tests for the ingest endpoint.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mobile_backend.offline.offline_analytics.routes import router
from mobile_backend.offline.offline_analytics.database import init_db
from mobile_backend.offline.offline_analytics.config import get_config
from datetime import datetime, timezone, timedelta

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup():
    init_db()
    cfg = get_config()
    # ensure SQLite DB created
    yield

def test_ingest_batch_minimal():
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "events": [
            {"device_id": "dev1", "user_id": "u1", "event_type": "open_app", "event_payload": {"screen": "home"}, "occurred_at": now},
            {"device_id": "dev1", "user_id": "u1", "event_type": "tap", "event_payload": {"x": 10, "y": 20}, "occurred_at": now},
        ]
    }
    r = client.post("/offline-analytics/ingest", json=payload)
    assert r.status_code == 201
    data = r.json()
    assert data["accepted"] == 2
