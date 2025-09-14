# Detects suspicious patterns and anomalies in security telemetry
"""
Security-focused anomaly detection utilities.

Security anomalies are typically sparse and high-impact; therefore:
 - IsolationForest is a good default for heterogeneous features (requires scikit-learn)
 - Z-score / EWMA still useful for high-volume numeric signals (e.g., auth fail rate)
 - Behavioral baselines (e.g., unusual geolocation for a user) are left as examples
"""

import numpy as np
from collections import deque
import logging
from typing import Dict, Any
from .config import SecurityMetricsConfig

logger = logging.getLogger("security_metrics.anomaly")

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not available; IsolationForest will be unavailable.")

class SecurityAnomalyDetector:
    """
    Simple multi-signal anomaly detector.

    - For numeric high-cardinality streams (auth failure counts per minute), use zscore/ewma.
    - For multi-feature events (ip reputation, geo mismatch, velocity), use IsolationForest when available.
    """

    def __init__(self, method: str = None, sensitivity: float = None):
        self.method = method or SecurityMetricsConfig.ANOMALY_METHOD
        self.sensitivity = sensitivity if sensitivity is not None else SecurityMetricsConfig.ANOMALY_SENSITIVITY
        self.history_numeric = deque(maxlen=1000)  # for zscore/ewma numeric streams

        # For vector-based detection
        self._vector_history = deque(maxlen=2000)
        self._if_model = IsolationForest(contamination=self.sensitivity, random_state=42) if (SKLEARN_AVAILABLE and self.method == "isolation_forest") else None

    # Numeric stream anomaly check (e.g., auth failures per minute)
    def numeric_check(self, value: float) -> bool:
        self.history_numeric.append(float(value))
        data = np.array(self.history_numeric, dtype=float)
        if data.size < 10:
            return False

        if self.method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            if std < 1e-8:
                return False
            z = abs((value - mean) / std)
            is_anom = z > (self.sensitivity * 10)  # sensitivity tuned differently for security
            logger.debug("numeric_check z=%s threshold=%s anom=%s", z, self.sensitivity, is_anom)
            return bool(is_anom)

        if self.method == "ewma":
            alpha = 0.2
            ewma = data[0]
            for v in data[1:]:
                ewma = alpha * v + (1 - alpha) * ewma
            deviation = abs(value - ewma)
            std = np.std(data)
            if std < 1e-8:
                return False
            is_anom = deviation > (self.sensitivity * 10 * std)
            logger.debug("numeric_check ewma dev=%s threshold=%s anom=%s", deviation, self.sensitivity, is_anom)
            return bool(is_anom)

        if self.method == "isolation_forest" and self._if_model:
            # fallback to vector approach if numeric-only used
            v = np.array(self.history_numeric).reshape(-1, 1)
            if v.shape[0] < 50:
                return False
            try:
                preds = self._if_model.fit_predict(v)
                return preds[-1] == -1
            except Exception:
                logger.exception("IsolationForest numeric predict failed")
                return False

        return False

    # Vector event anomaly check (multi-feature)
    def vector_check(self, vector: Dict[str, Any]) -> bool:
        """
        Accepts a mapping of features -> numeric values (e.g., {'country_distance': 1200, 'failed_logins_5m': 8, ...})
        Vector is normalized internally (simple).
        """
        # convert to sorted vector for deterministic encodings
        try:
            keys = sorted(vector.keys())
            arr = np.array([float(vector[k]) for k in keys], dtype=float)
        except Exception:
            logger.debug("vector_check: invalid vector")
            return False

        self._vector_history.append(arr)
        if self.method != "isolation_forest" or not SKLEARN_AVAILABLE:
            return False

        data = np.vstack(list(self._vector_history))
        if data.shape[0] < 50:
            return False
        try:
            preds = self._if_model.fit_predict(data)
            is_anom = preds[-1] == -1
            logger.debug("vector_check isolation forest result=%s", is_anom)
            return bool(is_anom)
        except Exception:
            logger.exception("IsolationForest vector predict failed")
            return False
