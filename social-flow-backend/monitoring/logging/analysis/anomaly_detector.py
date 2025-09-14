# Detects anomalies in logs using ML/statistical models
# monitoring/logging/analysis/anomaly_detector.py
"""
Anomaly detection on parsed logs.
Combines statistical z-score detection with ML-based Isolation Forest.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from .config import CONFIG


class AnomalyDetector:
    def __init__(self):
        self.z_threshold = CONFIG["ANOMALY_DETECTOR"]["zscore_threshold"]
        self.iforest = IsolationForest(**CONFIG["ANOMALY_DETECTOR"]["isolation_forest"])
        self.trained = False

    def fit(self, features: np.ndarray):
        """Fit Isolation Forest model on feature matrix."""
        self.iforest.fit(features)
        self.trained = True

    def detect(self, features: np.ndarray) -> dict:
        """Detect anomalies using hybrid method."""
        result = {"zscore_anomalies": [], "iforest_anomalies": []}

        # Z-score detection
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        zscores = np.abs((features - mean) / (std + 1e-9))
        z_anomalies = np.where(zscores > self.z_threshold)
        result["zscore_anomalies"] = list(zip(*z_anomalies))

        # Isolation forest detection
        if self.trained:
            preds = self.iforest.predict(features)
            result["iforest_anomalies"] = np.where(preds == -1)[0].tolist()

        return result
