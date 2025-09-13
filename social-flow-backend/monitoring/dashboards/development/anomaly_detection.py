# Anomaly detection logic for development metrics
"""
Anomaly Detection Engine

Uses rolling statistical methods + z-score to detect anomalies in metric streams.
"""

import statistics
from typing import List, Dict


class AnomalyDetector:
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold

    def detect(self, metric_name: str, values: List[float]) -> Dict[str, float]:
        if len(values) < 3:
            return {}

        mean = statistics.mean(values)
        stdev = statistics.pstdev(values)
        anomalies = {}

        for idx, v in enumerate(values):
            if stdev > 0:
                z = (v - mean) / stdev
                if abs(z) > self.z_threshold:
                    anomalies[f"{metric_name}[{idx}]"] = v

        return anomalies
