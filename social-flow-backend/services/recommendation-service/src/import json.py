import json
import os
import time
from typing import Dict, Any, Optional, List

REGISTRY_PATH = os.environ.get("ML_REGISTRY_PATH", os.path.join(os.path.dirname(__file__), "registry.json"))


def _ensure_registry():
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump({"models": []}, f)


def register_model(name: str, path: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register a model artifact in the registry and return the metadata record.
    """
    _ensure_registry()
    with open(REGISTRY_PATH, "r+", encoding="utf-8") as f:
        data = json.load(f)
        meta = {
            "id": f"{name}-{int(time.time())}",
            "name": name,
            "path": path,
            "metrics": metrics,
            "created_at": int(time.time()),
        }
        data.setdefault("models", []).append(meta)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    return meta


def latest_model(name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    _ensure_registry()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    models: List[Dict[str, Any]] = data.get("models", [])
    if name:
        models = [m for m in models if m.get("name") == name]
    if not models:
        return None
    # return the latest by created_at
    return sorted(models, key=lambda m: m.get("created_at", 0), reverse=True)[0]
