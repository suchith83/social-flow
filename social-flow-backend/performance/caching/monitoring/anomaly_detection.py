import numpy as np
from typing import Dict, Any

class CacheAnomalyDetector:
    """
    Detects anomalies in cache metrics using statistical thresholds (Z-score).
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.history = { "hit_ratio": [], "avg_latency_ms": [], "memory_usage_percent": [] }

    def update_history(self, metrics: Dict[str, Any]):
        """Add metrics to history for anomaly detection."""
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])
                if len(self.history[key]) > 500:
                    self.history[key].pop(0)

    def detect_anomalies(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Return anomaly flags for metrics."""
        anomalies = {}
        for key, values in self.history.items():
            if len(values) > 10:
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    z_score = (metrics[key] - mean) / std
                    anomalies[key] = abs(z_score) > self.threshold
        return anomalies
