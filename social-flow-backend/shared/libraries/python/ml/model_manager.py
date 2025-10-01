# common/libraries/python/ml/model_manager.py
"""
Model management: save, load, versioning.
Supports joblib, PyTorch, TensorFlow.
"""

import os
import joblib
from datetime import datetime
from .config import MLConfig

try:
    import torch
    import tensorflow as tf
except ImportError:
    torch = None
    tf = None

def save_model(model, name: str, fmt: str = None):
    fmt = fmt or MLConfig.MODEL_FORMAT
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MLConfig.MODEL_DIR, f"{name}_{timestamp}.{fmt}")
    os.makedirs(MLConfig.MODEL_DIR, exist_ok=True)

    if fmt == "joblib":
        joblib.dump(model, path)
    elif fmt == "torch" and torch:
        torch.save(model.state_dict(), path)
    elif fmt == "tf" and tf:
        model.save(path)
    else:
        raise ValueError(f"Unsupported format {fmt}")
    return path

def load_model(path: str, model_class=None, fmt: str = None):
    fmt = fmt or path.split(".")[-1]
    if fmt == "joblib":
        return joblib.load(path)
    elif fmt == "torch" and torch and model_class:
        model = model_class()
        model.load_state_dict(torch.load(path))
        return model
    elif fmt == "tf" and tf:
        return tf.keras.models.load_model(path)
    else:
        raise ValueError(f"Unsupported format {fmt}")
