"""
Audio Analysis Package Initialization

This module makes the audio-analysis package importable.
It loads configuration and provides easy access to core classes.
"""

from .config import AudioConfig
from .preprocessing import AudioPreprocessor
from .feature_extraction import FeatureExtractor
from .models import AudioClassifier, load_pretrained_model
from .inference import InferenceEngine
from .evaluation import Evaluator
from .pipeline import AudioPipeline

__all__ = [
    "AudioConfig",
    "AudioPreprocessor",
    "FeatureExtractor",
    "AudioClassifier",
    "load_pretrained_model",
    "InferenceEngine",
    "Evaluator",
    "AudioPipeline",
]
