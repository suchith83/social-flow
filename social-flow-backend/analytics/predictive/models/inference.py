"""
Inference utilities:
 - Load model pipeline and run batch or single predictions
 - Provide convenience batching, concurrency-safe prediction helpers
"""

from typing import Any, Sequence, List
import pandas as pd
import numpy as np
import os
from .utils import load_pickle, logger
from .model_registry import get_model
from .config import settings


def load_model_artifact(name: str, version: int | None = None):
    """
    Load a model artifact by registry name (and optional version).
    Returns the deserialized pipeline.
    """
    entry = get_model(name, version)
    path = entry["artifact_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = load_pickle(path)
    logger.info(f"Loaded model artifact {path}")
    return model


def predict_batch(model_pipeline: Any, X: pd.DataFrame, batch_size: int = 1000) -> List[Any]:
    """
    Batch predictions with a trained pipeline. Returns Python list of predictions.
    """
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        chunk = X.iloc[i : i + batch_size]
        chunk_preds = model_pipeline.predict(chunk)
        preds.extend(chunk_preds.tolist() if hasattr(chunk_preds, "tolist") else list(chunk_preds))
    logger.info(f"Produced {len(preds)} predictions for batch size {n}")
    return preds


def predict_one(model_pipeline: Any, x: dict):
    """
    Single-record prediction (accepts dict)
    """
    X = pd.DataFrame([x])
    pred = model_pipeline.predict(X)
    return float(pred[0]) if hasattr(pred, "__len__") else float(pred)
