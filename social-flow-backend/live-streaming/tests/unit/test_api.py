from fastapi.testclient import TestClient
from live_streaming.src.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_start_stop_stream():
    payload = {"title": "Test Stream", "uploader_id": "user_demo"}
    r = client.post("/streams/", json=payload)
    assert r.status_code == 201
    data = r.json()
    sid = data["id"]
    assert "ingest_url" in data and "playback_url" in data

    r2 = client.post(f"/streams/{sid}/start")
    assert r2.status_code == 200
    assert r2.json()["status"] == "live"

    r3 = client.post(f"/streams/{sid}/stop")
    assert r3.status_code == 200
    assert r3.json()["status"] == "stopped"
