"""NSFW detection local stub.

Provides:
- load_model(config=None) -> model
- model.predict(image_bytes) -> {"label": "safe"|"nsfw", "score": float}
"""
from typing import Any, Dict


def load_model(config: dict = None):
    class NSFWModel:
        def __init__(self, cfg=None):
            self.cfg = cfg or {}

        def predict(self, image_bytes: Any) -> Dict[str, Any]:
            # Deterministic local stub: always "safe" with high confidence unless override.
            return {"label": "safe", "score": 0.99}

    return NSFWModel(config)
