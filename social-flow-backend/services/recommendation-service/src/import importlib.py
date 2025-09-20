import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient
import time
import json


def load_app_from_path(py_path: Path, attr: str = "app"):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return getattr(module, attr)


SERVICE_MAIN = Path(__file__).resolve().parents[3] / "services" / "recommendation-service" / "src" / "main.py"


def test_feedback_then_recommendations_flow():
    app = load_app_from_path(SERVICE_MAIN)
    client = TestClient(app)

    # send feedback
    payload = {"user_id": "integration_user", "item_id": "video_42", "action": "like", "timestamp": int(time.time())}
    r = client.post("/feedback", json=payload)
    assert r.status_code == 200
    assert r.json().get("status") == "recorded"

    # small delay to simulate propagation in real systems (no-op here)
    time.sleep(0.1)

    # request recommendations to ensure service still responds correctly
    r = client.get("/recommendations/integration_user?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "integration_user"
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == 5
