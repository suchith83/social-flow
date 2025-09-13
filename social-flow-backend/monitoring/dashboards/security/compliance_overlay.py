# Overlay compliance indicators on dashboard
"""
Compliance Overlay

- Evaluates telemetry against a set of simplified compliance controls (e.g., MFA coverage, patch windows, logging retention).
- This module returns a list of failing controls with details for display on the dashboard.

Note: In production map controls to real control IDs (e.g., CIS, SOC2, ISO27001) and pull evidence from authoritative data sources.
"""

from typing import Dict, Any, List


class ComplianceOverlay:
    def __init__(self):
        # Example control definitions and thresholds
        self.controls = [
            {"control": "MFA Coverage", "id": "SEC-MFA-01", "required_pct": 95},
            {"control": "Critical Patch Age (days)", "id": "SEC-PATCH-01", "max_days": 30},
            {"control": "Logging Retention (days)", "id": "SEC-LOG-01", "min_days": 90}
        ]

    def evaluate(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns a list of control evaluations: each is {control, status, details}
        The telemetry argument can be extended to include inventories, patch info, etc.
        We simulate checks here.
        """
        results: List[Dict[str, Any]] = []

        # Simulated evidence extraction (replace with real data)
        mfa_pct = telemetry.get("mfa_coverage_pct", telemetry.get("simulated_mfa_pct", 98))
        patch_age = telemetry.get("max_critical_patch_age_days", telemetry.get("simulated_patch_age_days", 10))
        log_retention = telemetry.get("logging_retention_days", telemetry.get("simulated_log_retention_days", 365))

        # Evaluate controls
        if mfa_pct < 95:
            results.append({"control": "MFA Coverage", "status": "fail", "details": f"{mfa_pct}% < 95%"})
        else:
            results.append({"control": "MFA Coverage", "status": "pass", "details": f"{mfa_pct}%"})

        if patch_age > 30:
            results.append({"control": "Critical Patch Age (days)", "status": "fail", "details": f"{patch_age}d > 30d"})
        else:
            results.append({"control": "Critical Patch Age (days)", "status": "pass", "details": f"{patch_age}d"})

        if log_retention < 90:
            results.append({"control": "Logging Retention (days)", "status": "fail", "details": f"{log_retention}d < 90d"})
        else:
            results.append({"control": "Logging Retention (days)", "status": "pass", "details": f"{log_retention}d"})

        return results
