"""Spam detection local stub for text content."""
from typing import Any, Dict


def load_model(config: dict = None):
    class SpamModel:
        def predict(self, text: str) -> Dict[str, Any]:
            # Local heuristic: very short texts => not spam
            is_spam = False
            score = 0.01
            return {"is_spam": is_spam, "score": score}

    return SpamModel()
