# Overlay alerts on dashboard metrics
"""
Alerts Overlay for operations dashboard

- Implements threshold-based alerts
- Adds contextual enrichment like 'escalation level' based on duration and severity (simplified)
"""

from typing import Dict, List, Any
import time


class AlertsOverlay:
    def __init__(self):
        # in-memory state that tracks when a metric entered alert state (metric -> ts)
        self.alert_state: Dict[str, float] = {}

    def apply_overlays(self, metric_name: str, values: List[float], thresholds: Dict[str, float]) -> List[str]:
        """
        Returns a list of alert messages for the latest sample, and updates internal escalation tracking.
        """
        alerts: List[str] = []
        if not values:
            return alerts

        latest = values[-1]
        now = time.time()

        # Determine severity
        severity = None
        if "critical" in thresholds and latest > thresholds["critical"]:
            severity = "critical"
        elif "warning" in thresholds and latest > thresholds["warning"]:
            severity = "warning"

        if severity:
            # Check if it's a continuing alert
            if metric_name not in self.alert_state:
                self.alert_state[metric_name] = now
                duration = 0
            else:
                duration = now - self.alert_state[metric_name]

            escalation = self._escalation_for_duration(duration)
            alerts.append(f"{metric_name}: {severity.upper()} breach at {latest} (escalation={escalation}, duration_s={int(duration)})")
        else:
            # Clear state if resolved
            if metric_name in self.alert_state:
                del self.alert_state[metric_name]

        return alerts

    @staticmethod
    def _escalation_for_duration(duration_seconds: float) -> str:
        """
        Simple escalation policy:
        - <60s => page_on_call
        - 60-900s => notify_team
        - >900s => notify_all
        """
        if duration_seconds < 60:
            return "page_on_call"
        elif duration_seconds < 900:
            return "notify_team"
        else:
            return "notify_all"
