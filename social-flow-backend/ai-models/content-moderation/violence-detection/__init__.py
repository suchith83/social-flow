"""Violence detection local stub."""
from typing import Any, Dict


def load_model(config: dict = None):
    class ViolenceModel:
        def predict(self, image_or_video: Any) -> Dict[str, Any]:
            return {"label": "non-violent", "score": 0.98}

    return ViolenceModel()
