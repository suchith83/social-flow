from fastapi.testclient import TestClient
from src.main import app
import time

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_query():
    now = time.time()
    payload = {"name": "views", "value": 1.0, "tags": {"video_id": "v1"}, "timestamp": now}
    r = client.post("/ingest", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ingested"
    assert j["name"] == "views"

    q = client.get(f"/metrics?name=views&start_ts={now-1}&end_ts={now+1}")
    assert q.status_code == 200
    data = q.json()
    assert data["name"] == "views"
    assert data["count"] >= 1
    assert isinstance(data["points"], list)
    assert any(p["tags"].get("video_id") == "v1" for p in data["points"])
