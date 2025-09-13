# Security dashboard main logic
"""
Security Dashboard Runner

- Loads config
- Pulls telemetry through SecurityAdapter
- Runs threat detection and compliance overlays
- Exposes quick CLI render and programmatic API for UI integration
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from monitoring.dashboards.security.security_adapter import SecurityAdapter
from monitoring.dashboards.security.threat_detection import ThreatDetector
from monitoring.dashboards.security.compliance_overlay import ComplianceOverlay
from monitoring.dashboards.security.incident_response_helper import IncidentResponseHelper


class SecurityDashboard:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.adapter = SecurityAdapter(self.config.get("backends", {}))
        self.detector = ThreatDetector()
        self.compliance = ComplianceOverlay()
        self.ir_helper = IncidentResponseHelper(self.config.get("response_playbooks", {}))
        self.refresh_interval = self.config.get("refresh_interval_seconds", 45)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def start(self) -> None:
        """
        Main loop. Designed to be robust: each iteration collects data, detects threats,
        applies compliance overlays, and optionally triggers playbooks (simulation only).
        """
        print(f"🔒 Starting {self.config.get('dashboard_name')} (refresh {self.refresh_interval}s)")
        try:
            while True:
                try:
                    lookback = self.config.get("lookback_minutes", 30)
                    telemetry = self.adapter.fetch_telemetry(lookback_minutes=lookback)
                    findings = self.detector.scan_bundle(telemetry)
                    compliance_issues = self.compliance.evaluate(telemetry)

                    # Render widgets
                    for w in self.config.get("widgets", []):
                        self._render_widget(w, telemetry, findings)

                    # Render summaries
                    self._render_findings_summary(findings)
                    self._render_compliance_summary(compliance_issues)

                    # Simulate (safe) auto-response decisions (no destructive actions)
                    self._simulate_playbooks(findings)

                except Exception as e:
                    # Prevent the loop from dying due to transient errors
                    print(f"[ERROR] security iteration failed: {e}")

                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("🛑 Shutdown requested. Exiting.")

    def _render_widget(self, widget: Dict[str, Any], telemetry: Dict[str, Any], findings: Dict[str, Any]) -> None:
        title = widget.get("title", widget.get("metric"))
        metric = widget.get("metric")
        print(f"--- {title} ---")
        if metric == "suspicious_ips":
            ips = telemetry.get("suspicious_ips", [])[:10]
            for ip in ips:
                enriched = self.adapter.enrich_ip(ip)
                print(f"{ip} - {enriched.get('country', 'unknown')} - hits={enriched.get('hits', 0)}")
        else:
            # Generic numeric widgets
            value = telemetry.get(metric)
            if value is None and metric in findings:
                value = findings.get(metric)
            print(f"Value: {value}")
            thresholds = widget.get("thresholds", {})
            if thresholds:
                # simple threshold check
                latest = value if isinstance(value, (int, float)) else (value[-1] if isinstance(value, list) and value else None)
                if latest is not None:
                    if "critical" in thresholds and latest > thresholds["critical"]:
                        print(f"🚨 CRITICAL: {latest} > {thresholds['critical']}")
                    elif "warning" in thresholds and latest > thresholds["warning"]:
                        print(f"⚠ WARNING: {latest} > {thresholds['warning']}")
        print("")

    def _render_findings_summary(self, findings: Dict[str, Any]) -> None:
        print("=== Threat Findings ===")
        for k, v in findings.items():
            print(f"- {k}: {v if isinstance(v, (str, int)) else len(v)}")
        print("")

    def _render_compliance_summary(self, issues: List[Dict[str, Any]]) -> None:
        print("=== Compliance Issues ===")
        if not issues:
            print("All checks passed.")
        else:
            for issue in issues:
                print(f"- {issue.get('control')}: {issue.get('status')} ({issue.get('details')})")
        print("")

    def _simulate_playbooks(self, findings: Dict[str, Any]) -> None:
        """
        In production this would create tickets / call playbooks with approval.
        Here we simulate safe decisions and log suggested actions.
        """
        # Pick suspicious IPs as candidates
        suspicious = findings.get("suspicious_ips", [])
        if suspicious:
            for ip in suspicious[:3]:
                decision = self.ir_helper.suggest_playbook_for_ip(ip)
                print(f"[PLAYBOOK SUGGESTION] ip={ip} -> {decision}")

    # Programmatic API (useful for UI backends)
    def get_latest_snapshot(self) -> Dict[str, Any]:
        lookback = self.config.get("lookback_minutes", 30)
        telemetry = self.adapter.fetch_telemetry(lookback_minutes=lookback)
        findings = self.detector.scan_bundle(telemetry)
        compliance_issues = self.compliance.evaluate(telemetry)
        return {
            "telemetry": telemetry,
            "findings": findings,
            "compliance": compliance_issues
        }


if __name__ == "__main__":
    sd = SecurityDashboard("monitoring/dashboards/security/sec_dashboard_config.json")
    sd.start()
