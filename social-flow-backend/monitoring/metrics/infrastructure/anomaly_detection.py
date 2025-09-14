# Detects anomalies or performance regressions in infra data
"""
Infrastructure anomaly detection utilities.

Supports:
 - EWMA-based anomaly detection for noisy infra metrics (CPU, memory spikes)
 - Z-score based (statistical)
 - IsolationForest for more complex patterns (requires scikit-learn)
"""

import numpy as np
from collections import deque
import logging
from .config import InfraMetricsConfig

logger = logging.getLogger("infra_metrics.anomaly")


class InfraAnomalyDetector:
    def __init__(self, window: int = 300, method: str = None, sensitivity: float = None):
        self.window = window
        self.method = method or InfraMetricsConfig.ANOMALY_METHOD
        self.sensitivity = sensitivity or InfraMetricsConfig.ANOMALY_SENSITIVITY
        self.history = deque(maxlen=self.window)
        self._model = None
        if self.method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
                self._model = IsolationForest(contamination=0.05, random_state=42)
            except Exception:
                logger.exception("IsolationForest requested but scikit-learn not available; falling back to zscore")
                self.method = "zscore"

    def update_and_check(self, value: float) -> bool:
        """
        Add a new value to history and return True if value is anomalous.
        Methods:
         - zscore: compares to mean/std (robust for normal-ish distributions)
         - ewma: tracks exponentially weighted mean and checks deviation
         - isolation_forest: ML based (requires sufficient history)
        """
        self.history.append(value)
        data = np.array(self.history, dtype=float)
        if data.size < 5:
            return False  # not enough data yet

        if self.method == "zscore":
            mean, std = np.mean(data), np.std(data)
            if std < 1e-8:
                return False
            z = abs((value - mean) / std)
            is_anom = z > self.sensitivity
            logger.debug("ZScore check val=%s mean=%s std=%s z=%s anom=%s", value, mean, std, z, is_anom)
            return bool(is_anom)

        if self.method == "ewma":
            alpha = 2.0 / (len(data) + 1)  # adaptive alpha
            ewma = data[0]
            for v in data[1:]:
                ewma = alpha * v + (1 - alpha) * ewma
            deviation = abs(value - ewma)
            std = np.std(data)
            if std < 1e-8:
                return False
            is_anom = deviation > (self.sensitivity * std)
            logger.debug("EWMA check val=%s ewma=%s std=%s dev=%s anom=%s", value, ewma, std, deviation, is_anom)
            return bool(is_anom)

        if self.method == "isolation_forest":
            if data.size < 50:
                return False
            # train and predict
            try:
                preds = self._model.fit_predict(data.reshape(-1, 1))
                is_anom = preds[-1] == -1
                logger.debug("IForest predict last=%s", preds[-1])
                return bool(is_anom)
            except Exception:
                logger.exception("IsolationForest prediction failed")
                return False

        return False
