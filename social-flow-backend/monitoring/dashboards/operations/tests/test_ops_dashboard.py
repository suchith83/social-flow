# Unit tests for operations dashboard
"""
Unit tests for operations dashboard components (infra adapter, incident analyzer, capacity planner, alerts)
"""

import unittest
from monitoring.dashboards.operations.infra_adapter import InfraAdapter, BackendFailure
from monitoring.dashboards.operations.incident_analysis import IncidentAnalyzer
from monitoring.dashboards.operations.capacity_planner import CapacityPlanner
from monitoring.dashboards.operations.alerts_overlay import AlertsOverlay


class TestOpsDashboardComponents(unittest.TestCase):
    def setUp(self):
        backends = {"prometheus": {"enabled": True}, "cloudwatch": {"enabled": False}}
        self.adapter = InfraAdapter(backends)
        self.analyzer = IncidentAnalyzer(z_spike_threshold=1.5)
        self.planner = CapacityPlanner(headroom_target=0.2)
        self.overlay = AlertsOverlay()

    def test_fetch_metric_success(self):
        series = self.adapter.fetch_metric("node_cpu_seconds_total:rate:avg", window_minutes=5, points=10)
        self.assertIsInstance(series, list)
        self.assertGreaterEqual(len(series), 1)

    def test_fetch_metric_circuit_breaker_behavior(self):
        # artificially mark the circuit as opened by recording failures
        for _ in range(6):
            self.adapter.circuit.record_failure()
        self.assertFalse(self.adapter.circuit.allowed())
        with self.assertRaises(BackendFailure):
            self.adapter.fetch_metric("any_metric", window_minutes=1)

    def test_incident_analyzer_detects_spike(self):
        # create a data bundle with a clear spike
        bundle = {
            "cpu": [10, 12, 11, 100],  # spike at end
            "memory": [60, 62, 59, 61],
            "latency": [5, 6, 7, 8]
        }
        result = self.analyzer.correlate_recent_incidents(bundle)
        self.assertIn("spike_count", result)
        self.assertGreaterEqual(result["spike_count"], 1)
        self.assertTrue(len(result["top_suspects"]) >= 1)

    def test_capacity_planner_recommendation(self):
        bundle = {
            "cpu": [10, 20, 30, 35, 40],  # trending up
            "memory": [60, 61, 60, 59, 58]
        }
        report = self.planner.estimate_capacity_requirements(bundle)
        self.assertIn("cpu", report)
        self.assertIn("memory", report)
        self.assertTrue(report["cpu"]["recommendation"] in ("scale_up", "no_change", "scale_down"))

    def test_alerts_overlay_escalation(self):
        thresholds = {"warning": 50, "critical": 80}
        # initial alert creation
        messages = self.overlay.apply_overlays("cpu", [30, 40, 85], thresholds)
        self.assertTrue(any("CRITICAL" in m or "CRITICAL" in m.upper() for m in messages))
        # subsequent call should preserve escalation duration > 0
        time.sleep(0.1)
        messages2 = self.overlay.apply_overlays("cpu", [40, 45, 90], thresholds)
        self.assertTrue(any("escalation=" in m for m in messages2))


if __name__ == "__main__":
    unittest.main()
