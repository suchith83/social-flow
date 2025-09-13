# Executive dashboard main logic
"""
Executive Dashboard Renderer

This module loads configuration, fetches KPIs using the KPI adapter,
applies trend analysis & SLA overlays, and renders high-level metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

from monitoring.dashboards.executive.kpi_adapter import KPIAdapter
from monitoring.dashboards.executive.trend_analysis import TrendAnalyzer
from monitoring.dashboards.executive.sla_overlay import SLAOverlay


class ExecDashboard:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.adapter = KPIAdapter()
        self.trend_analyzer = TrendAnalyzer()
        self.sla_overlay = SLAOverlay()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def render(self) -> None:
        refresh_interval = self.config.get("refresh_interval_seconds", 60)
        print(f"📊 Starting {self.config['dashboard_name']} with refresh {refresh_interval}s")

        while True:
            for widget in self.config.get("widgets", []):
                self._render_widget(widget)

            print("✅ Executive Dashboard Refresh Complete\n")
            time.sleep(refresh_interval)

    def _render_widget(self, widget: Dict[str, Any]) -> None:
        metric = widget["metric"]
        values = self.adapter.fetch_kpi(metric)

        trend = self.trend_analyzer.analyze(metric, values)
        sla_status = self.sla_overlay.evaluate(metric, values, widget.get("thresholds", {}))

        print(f"📌 {widget['title']} ({metric})")
        print(f"   ➡ Latest: {values[-1] if values else 'N/A'}")
        if trend:
            print(f"   📈 Trend: {trend}")
        if sla_status:
            print(f"   🚨 SLA Breach: {sla_status}")


if __name__ == "__main__":
    dashboard = ExecDashboard("monitoring/dashboards/executive/exec_dashboard_config.json")
    dashboard.render()
