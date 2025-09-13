# Unit tests for sync-engine
"""
Basic tests for sync engine push & pull flows.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mobile_backend.offline.sync_engine.routes import router
from mobile_backend.offline.sync_engine.database import init_db, SessionLocal
from datetime import datetime, timezone
from mobile_backend.offline.sync_engine.models import PushBatchRequest, PushOperation, ChangeType, ItemPayload

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    yield


def test_push_and_pull_roundtrip():
    # push a create op
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "client_id": "client1",
        "last_sync_seq": 0,
        "ops": [
            {
                "type": "create",
                "item": {
                    "key": "note:1",
                    "data": {"text": "hello"},
                    "client_version": 1
                }
            }
        ]
    }
    r = client.post("/sync/push", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert res["applied"] == 1
    last_seq = res["last_seq"]

    # pull changes since 0
    pull_payload = {"client_id": "client1", "since_seq": 0, "page_size": 100}
    p = client.post("/sync/pull", json=pull_payload)
    assert p.status_code == 200
    data = p.json()
    assert "changes" in data
    assert len(data["changes"]) >= 1
    assert data["last_seq"] >= last_seq
