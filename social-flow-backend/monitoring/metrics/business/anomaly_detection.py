# Detects anomalies or unusual trends in business metrics
"""
Anomaly detection for business KPIs.
"""

import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from .config import BusinessMetricsConfig


class BusinessAnomalyDetector:
    def __init__(self, method: str = BusinessMetricsConfig.ANOMALY_METHOD):
        self.method = method
        self.history = deque(maxlen=500)
        if method == "isolation_forest":
            self.model = IsolationForest(contamination=0.1, random_state=42)
        else:
            self.model = None

    def update(self, value: float) -> bool:
        self.history.append(value)
        data = np.array(self.history)

        if self.method == "zscore":
            mean, std = np.mean(data), np.std(data)
            return abs((value - mean) / (std + 1e-8)) > BusinessMetricsConfig.ANOMALY_SENSITIVITY

        elif self.method == "ewma":
            alpha = 0.3
            ewma = data[0]
            for v in data[1:]:
                ewma = alpha * v + (1 - alpha) * ewma
            return abs(value - ewma) > BusinessMetricsConfig.ANOMALY_SENSITIVITY * np.std(data)

        elif self.method == "isolation_forest" and len(data) > 20:
            preds = self.model.fit_predict(data.reshape(-1, 1))
            return preds[-1] == -1

        return False
