"""
Scene Detection Package Initialization
"""

from .config import SceneConfig
from .preprocessing import VideoPreprocessor
from .feature_extraction import FeatureExtractor
from .detection import SceneDetector
from .models import SceneClassifier, load_pretrained_model
from .inference import InferenceEngine
from .evaluation import Evaluator
from .pipeline import ScenePipeline

__all__ = [
    "SceneConfig",
    "VideoPreprocessor",
    "FeatureExtractor",
    "SceneDetector",
    "SceneClassifier",
    "load_pretrained_model",
    "InferenceEngine",
    "Evaluator",
    "ScenePipeline",
]

"""Scene detection local stub."""
from typing import Any, Dict, List


def load_model(config: dict = None):
    class SceneModel:
        def predict(self, image_or_video) -> List[Dict[str, Any]]:
            return [{"scene": "outdoor", "confidence": 0.9}]

    return SceneModel()
