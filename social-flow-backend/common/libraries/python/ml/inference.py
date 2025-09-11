# common/libraries/python/ml/inference.py
"""
Inference utilities: batch + real-time.
"""

import numpy as np
from typing import Any

def batch_inference(model, X: Any):
    """Run inference on a batch of data."""
    return model.predict(X)

def real_time_inference(model, x: Any):
    """Run inference on a single input."""
    arr = np.array([x]) if not isinstance(x, (list, np.ndarray)) else np.array(x)
    return model.predict(arr)[0]
