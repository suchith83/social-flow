# Unit tests for development dashboard
"""
Unit Tests for Development Dashboard
"""

import unittest
from monitoring.dashboards.development.metrics_adapter import MetricsAdapter
from monitoring.dashboards.development.anomaly_detection import AnomalyDetector
from monitoring.dashboards.development.alerts_overlay import AlertsOverlay


class TestDevDashboard(unittest.TestCase):
    def setUp(self):
        self.adapter = MetricsAdapter()
        self.detector = AnomalyDetector()
        self.overlay = AlertsOverlay()

    def test_fetch_metric(self):
        data = self.adapter.fetch_metric("http_request_duration_seconds", 5)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(isinstance(v, float) for v in data))

    def test_anomaly_detection(self):
        values = [100, 102, 98, 500]  # One anomaly
        anomalies = self.detector.detect("latency", values)
        self.assertIn("latency[3]", anomalies)

    def test_alerts_overlay(self):
        values = [100, 200, 600]
        thresholds = {"warning": 200, "critical": 500}
        alerts = self.overlay.apply_overlays("latency", values, thresholds)
        self.assertIn("latency: CRITICAL breach at 600", alerts)


if __name__ == "__main__":
    unittest.main()
