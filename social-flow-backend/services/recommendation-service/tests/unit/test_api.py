from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recommendations():
    r = client.get("/recommendations/test_user?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "test_user"
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == 5


def test_feedback_endpoint():
    payload = {
        "user_id": "test_user",
        "item_id": "video_1",
        "action": "view",
        "timestamp": 1690000000,
    }
    r = client.post("/feedback", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "recorded"
