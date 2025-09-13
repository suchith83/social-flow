# Unit tests for security dashboard
"""
Unit tests for security dashboard components
"""

import unittest
from monitoring.dashboards.security.security_adapter import SecurityAdapter, BackendError
from monitoring.dashboards.security.threat_detection import ThreatDetector
from monitoring.dashboards.security.compliance_overlay import ComplianceOverlay
from monitoring.dashboards.security.incident_response_helper import IncidentResponseHelper


class TestSecurityDashboardComponents(unittest.TestCase):
    def setUp(self):
        backends = {"siem": {"enabled": True}, "ids": {"enabled": True}, "vuln_scanner": {"enabled": False}}
        self.adapter = SecurityAdapter(backends)
        self.detector = ThreatDetector()
        self.compliance = ComplianceOverlay()
        self.ir = IncidentResponseHelper({"suspicious_ip": "isolate_host && enrich_whois && notify_security_team", "brute_force": "block_ip && notify_oncall"})

    def test_fetch_telemetry_contains_keys(self):
        telemetry = self.adapter.fetch_telemetry(lookback_minutes=10)
        self.assertIn("failed_logins_per_minute", telemetry)
        self.assertIn("suspicious_ips", telemetry)

    def test_enrich_ip_returns_structure(self):
        ip = "198.51.100.5"
        info = self.adapter.enrich_ip(ip)
        self.assertIn("country", info)
        self.assertIn("hits", info)

    def test_threat_detector_detects_bruteforce_candidates(self):
        telemetry = {
            "auth.failed_logins": [1, 2, 3, 400],
            "suspicious_ip_details": [{"ip": "198.51.100.5", "hits": 200}],
            "suspicious_ips": ["198.51.100.5"]
        }
        findings = self.detector.scan_bundle(telemetry)
        self.assertIn("brute_force_candidates", findings)
        self.assertTrue(len(findings["brute_force_candidates"]) >= 1)

    def test_compliance_overlay_reports(self):
        telemetry = {"simulated_mfa_pct": 90, "simulated_patch_age_days": 40, "simulated_log_retention_days": 30}
        issues = self.compliance.evaluate(telemetry)
        self.assertTrue(any(i["status"] == "fail" for i in issues))

    def test_incident_response_suggests_playbook(self):
        suggestion = self.ir.suggest_playbook_for_ip("198.51.100.5")
        self.assertIn("tokens", suggestion)
        self.assertIn("simulated_actions", suggestion)

    def test_adapter_handles_transient_errors(self):
        # Simulate multiple calls to surface potential transient errors — should not raise uncaught
        for _ in range(5):
            try:
                _ = self.adapter.fetch_telemetry(lookback_minutes=1)
            except BackendError:
                # acceptable, adapter may raise when simulating errors
                pass


if __name__ == "__main__":
    unittest.main()
