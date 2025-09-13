# Development dashboard main logic
"""
Development Dashboard Renderer

This module loads configuration, fetches metrics using the metrics adapter,
and renders a developer-focused dashboard with anomaly overlays and alerts.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from monitoring.dashboards.development.metrics_adapter import MetricsAdapter
from monitoring.dashboards.development.anomaly_detection import AnomalyDetector
from monitoring.dashboards.development.alerts_overlay import AlertsOverlay


class DevDashboard:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.adapter = MetricsAdapter()
        self.anomaly_detector = AnomalyDetector()
        self.overlay = AlertsOverlay()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def render(self) -> None:
        """
        Continuously refresh the dashboard with updated metrics.
        """
        refresh_interval = self.config.get("refresh_interval_seconds", 30)
        print(f"🚀 Starting {self.config['dashboard_name']} with refresh {refresh_interval}s")

        while True:
            widgets = self.config.get("widgets", [])
            for widget in widgets:
                self._render_widget(widget)

            print("✅ Refresh complete\n")
            time.sleep(refresh_interval)

    def _render_widget(self, widget: Dict[str, Any]) -> None:
        metric = widget["metric"]
        data = self.adapter.fetch_metric(metric)

        anomalies = self.anomaly_detector.detect(metric, data)
        alerts = self.overlay.apply_overlays(metric, data, widget.get("thresholds", {}))

        print(f"📊 {widget['title']} [{metric}]")
        print(f"   ➡ Latest Value: {data[-1] if data else 'N/A'}")
        if anomalies:
            print(f"   ⚠ Anomalies: {anomalies}")
        if alerts:
            print(f"   🚨 Alerts: {alerts}")


if __name__ == "__main__":
    dashboard = DevDashboard("monitoring/dashboards/development/dev_dashboard_config.json")
    dashboard.render()
