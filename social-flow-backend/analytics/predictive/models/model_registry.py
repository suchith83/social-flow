"""
Simple Model Registry:
 - Local registry storing metadata + artifact path
 - Optionally integrate with MLflow if settings.MLFLOW_URI is set
"""

import os
import json
from typing import Dict, Any
from .utils import ensure_dir, save_pickle, load_pickle, to_json, timestamp, logger
from .config import settings

REGISTRY_PATH = os.path.join(settings.MODEL_DIR, "registry.json")


def _ensure_registry():
    ensure_dir(settings.MODEL_DIR)
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump({"models": []}, f)


def register_model(name: str, artifact_path: str, metrics: Dict[str, Any], metadata: Dict[str, Any] | None = None):
    """
    Register a model in the local registry. Returns registry entry object.
    """
    _ensure_registry()
    with open(REGISTRY_PATH, "r+", encoding="utf-8") as f:
        registry = json.load(f)
        version = 1 + max([m.get("version", 0) for m in registry["models"] if m["name"] == name], default=0)
        entry = {
            "name": name,
            "version": version,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": timestamp(),
        }
        registry["models"].append(entry)
        f.seek(0)
        json.dump(registry, f, indent=2)
        f.truncate()
    logger.info(f"Registered model {name} v{version}")
    return entry


def list_models(name: str | None = None):
    _ensure_registry()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    if name:
        return [m for m in registry["models"] if m["name"] == name]
    return registry["models"]


def get_model(name: str, version: int | None = None) -> Dict[str, Any]:
    _ensure_registry()
    models = list_models(name)
    if not models:
        raise KeyError(f"No models found with name {name}")
    if version is None:
        # return latest
        models_sorted = sorted(models, key=lambda m: m["version"], reverse=True)
        return models_sorted[0]
    for m in models:
        if m["version"] == version:
            return m
    raise KeyError(f"Model {name} v{version} not found")
