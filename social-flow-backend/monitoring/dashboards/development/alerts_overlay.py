# Overlay alerts on dashboard metrics
"""
Alerts Overlay

Applies threshold-based alerting to metrics for dashboard overlays.
"""

from typing import Dict, List


class AlertsOverlay:
    def apply_overlays(self, metric_name: str, values: List[float], thresholds: Dict[str, float]) -> List[str]:
        alerts = []
        if not values:
            return alerts

        latest = values[-1]

        if "critical" in thresholds and latest > thresholds["critical"]:
            alerts.append(f"{metric_name}: CRITICAL breach at {latest}")
        elif "warning" in thresholds and latest > thresholds["warning"]:
            alerts.append(f"{metric_name}: WARNING breach at {latest}")

        return alerts
