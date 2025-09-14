# Tests for anomaly_detector
# monitoring/logging/analysis/tests/test_anomaly_detector.py
import numpy as np
from monitoring.logging.analysis.anomaly_detector import AnomalyDetector

def test_anomaly_detection():
    X = np.random.normal(0, 1, (100, 2))
    detector = AnomalyDetector()
    detector.fit(X)
    results = detector.detect(X)
    assert "iforest_anomalies" in results
