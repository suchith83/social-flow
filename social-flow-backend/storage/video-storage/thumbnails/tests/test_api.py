import pytest
from fastapi.testclient import TestClient
from thumbnails.api import app

client = TestClient(app)

def test_generate_endpoint_queue():
    payload = {
        "video_id": "v1",
        "video_path": "/tmp/nonexistent.mp4",
        "timestamps": [1.0,2.0],
        "count": 2,
        "sizes": ["320x180"],
        "formats": ["jpeg"],
        "async_job": True
    }
    r = client.post("/thumbnails/generate", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["video_id"] == "v1"
    assert j["thumbnails"] == []
