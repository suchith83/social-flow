# Detects anomalies in traffic/performance
# performance/cdn/analytics/anomaly_detector.py
"""
Anomaly Detector
================
Detects anomalies in CDN performance using statistical methods
and rolling window analysis.
"""

from collections import deque
import statistics
from typing import Deque, Dict
from .utils import logger

class AnomalyDetector:
    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.latency_window: Deque[float] = deque(maxlen=window_size)

    def check(self, record: Dict) -> bool:
        """Check if the given record is anomalous."""
        latency = record.get("latency_ms")
        if latency is None:
            return False

        self.latency_window.append(latency)
        if len(self.latency_window) < self.window_size:
            return False  # not enough data

        mean = statistics.mean(self.latency_window)
        stdev = statistics.pstdev(self.latency_window)
        if stdev == 0:
            return False

        z_score = abs((latency - mean) / stdev)
        if z_score > self.threshold:
            logger.warning(f"Anomaly detected! latency={latency}, z={z_score:.2f}")
            return True
        return False
