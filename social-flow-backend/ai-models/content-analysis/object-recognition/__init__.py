"""
Object Recognition Package Initialization

Provides access to core components of the object recognition pipeline.
"""

from .config import ObjectConfig
from .preprocessing import ImagePreprocessor
from .feature_extraction import FeatureExtractor
from .models import ObjectRecognizer, load_pretrained_model
from .inference import InferenceEngine
from .evaluation import Evaluator
from .pipeline import ObjectPipeline

__all__ = [
    "ObjectConfig",
    "ImagePreprocessor",
    "FeatureExtractor",
    "ObjectRecognizer",
    "load_pretrained_model",
    "InferenceEngine",
    "Evaluator",
    "ObjectPipeline",
]
