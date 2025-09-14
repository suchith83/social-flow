# Detects anomalies in application metrics
"""
Anomaly detection on application metrics.
Supports Z-score, EWMA (Exponential Weighted Moving Average), and Isolation Forest.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from .config import MetricsConfig


class AnomalyDetector:
    """Detects anomalies in metrics."""

    def __init__(self, method: str = MetricsConfig.ANOMALY_DETECTION_METHOD):
        self.method = method
        self.history = deque(maxlen=500)  # keep last 500 values
        if method == "isolation_forest":
            self.model = IsolationForest(contamination=0.05, random_state=42)
        else:
            self.model = None

    def update(self, value: float) -> bool:
        """Update with new value and return whether it's anomalous."""
        self.history.append(value)
        data = np.array(self.history)

        if self.method == "zscore":
            mean, std = np.mean(data), np.std(data)
            return abs((value - mean) / (std + 1e-8)) > MetricsConfig.ANOMALY_SENSITIVITY

        elif self.method == "ewma":
            alpha = 0.3
            ewma = data[0]
            for v in data[1:]:
                ewma = alpha * v + (1 - alpha) * ewma
            return abs(value - ewma) > MetricsConfig.ANOMALY_SENSITIVITY * np.std(data)

        elif self.method == "isolation_forest" and len(data) > 20:
            preds = self.model.fit_predict(data.reshape(-1, 1))
            return preds[-1] == -1

        return False
