"""Collaborative filtering local stub: returns simple ranked item ids."""
from typing import Any, Dict, List


def load_model(config: dict = None):
    class CFModel:
        def predict(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
            return [{"item_id": f"cf_item_{i}", "score": 1.0 / (i + 1)} for i in range(limit)]

    return CFModel()
