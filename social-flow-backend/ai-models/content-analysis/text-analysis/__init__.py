"""Text analysis local stub."""
from typing import Any, Dict


def load_model(config: dict = None):
    class TextModel:
        def predict(self, text: str) -> Dict[str, Any]:
            return {"sentiment": "neutral", "entities": []}

    return TextModel()
