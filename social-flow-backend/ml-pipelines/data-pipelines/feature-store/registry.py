# Feature metadata registry
# registry.py
import json
from typing import Dict, Any
from pathlib import Path
from .utils import logger

class FeatureRegistry:
    """
    Stores metadata about available features and their schemas.
    """

    def __init__(self, path: str = "feature_registry.json"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text(json.dumps({"features": {}}))

    def register(self, name: str, schema: Dict[str, Any], description: str = ""):
        registry = self._load()
        registry["features"][name] = {"schema": schema, "description": description}
        self._save(registry)
        logger.info(f"Registered feature: {name}")

    def list_features(self) -> Dict[str, Any]:
        return self._load()["features"]

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    def _save(self, data: Dict[str, Any]):
        self.path.write_text(json.dumps(data, indent=2))
