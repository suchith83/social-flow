# common/libraries/python/ml/config.py
"""
Configuration for ML library.
"""

import os

class MLConfig:
    DATA_DIR = os.getenv("ML_DATA_DIR", "./data")
    MODEL_DIR = os.getenv("ML_MODEL_DIR", "./models")
    EXPERIMENTS_DIR = os.getenv("ML_EXPERIMENTS_DIR", "./experiments")

    # Default model format (joblib, torch, tf)
    MODEL_FORMAT = os.getenv("ML_MODEL_FORMAT", "joblib")
