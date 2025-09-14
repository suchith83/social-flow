# scripts/monitoring/dashboard_generator.py
import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("monitoring.dashboard")


class DashboardGenerator:
    """
    Emit a simple Grafana dashboard JSON file based on configured checks and metrics.
    This is intentionally small: generate panels for synthetic checks and host metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("monitoring", {}).get("dashboard", {}).get("output_dir", "./dashboards")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self) -> str:
        checks = self.config.get("monitoring", {}).get("synthetic", {}).get("checks", [])
        panels = []
        panel_id = 1

        # Panel for host CPU
        panels.append({
            "id": panel_id,
            "title": "Host CPU %",
            "type": "graph",
            "targets": [{"expr": "host_cpu_percent"}],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        })
        panel_id += 1

        # Panels for each synthetic check latency
        for i, c in enumerate(checks):
            panels.append({
                "id": panel_id,
                "title": f"Latency: {c.get('name')}",
                "type": "graph",
                "targets": [{"expr": f"synthetic_check_latency_seconds{{check=\"{c.get('name')}\"}}"}],
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 8 + (i * 6)}
            })
            panel_id += 1

        dashboard = {
            "dashboard": {
                "title": "SocialFlow Monitoring",
                "panels": panels,
                "schemaVersion": 16
            },
            "overwrite": True
        }

        out_path = os.path.join(self.output_dir, "socialflow-monitor-dashboard.json")
        with open(out_path, "w") as fh:
            json.dump(dashboard, fh, indent=2)
        logger.info("Wrote dashboard to %s", out_path)
        return out_path
