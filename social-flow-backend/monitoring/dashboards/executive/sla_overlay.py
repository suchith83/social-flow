# Overlay SLA compliance indicators on dashboard
"""
SLA Overlay

Evaluates metrics against SLA thresholds and highlights breaches.
"""

from typing import Dict, List, Optional


class SLAOverlay:
    def evaluate(self, metric_name: str, values: List[float], thresholds: Dict[str, float]) -> Optional[str]:
        if not values:
            return None

        latest = values[-1]

        if "critical" in thresholds and latest < thresholds["critical"]:
            return f"{metric_name}: CRITICAL SLA breach ({latest})"
        elif "warning" in thresholds and latest < thresholds["warning"]:
            return f"{metric_name}: WARNING SLA risk ({latest})"

        return None
