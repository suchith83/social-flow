# Operations dashboard main logic
"""
Operations Dashboard Renderer (synchronous driver with async adapter support)

Responsibilities:
- Load configuration
- Coordinate fetching metrics (via InfraAdapter)
- Run incident correlation & capacity forecasts
- Apply overlays / alerts
- Provide an extensible render hook (print to stdout by default)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from monitoring.dashboards.operations.infra_adapter import InfraAdapter
from monitoring.dashboards.operations.incident_analysis import IncidentAnalyzer
from monitoring.dashboards.operations.capacity_planner import CapacityPlanner
from monitoring.dashboards.operations.alerts_overlay import AlertsOverlay


class OpsDashboard:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        # InfraAdapter may use async internals; we use blocking wrappers here for simplicity.
        self.adapter = InfraAdapter(self.config.get("backends", {}))
        self.analyzer = IncidentAnalyzer()
        self.planner = CapacityPlanner()
        self.overlay = AlertsOverlay()
        self.refresh_interval = self.config.get("refresh_interval_seconds", 20)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def start(self) -> None:
        """
        Main loop. Designed to be robust: exceptions during a single widget won't stop the loop.
        For production use consider running each widget in separate processes/containers.
        """
        print(f"⚙️  Starting {self.config.get('dashboard_name')} (refresh {self.refresh_interval}s)")
        try:
            while True:
                widgets = self.config.get("widgets", [])
                # Fetch all metrics in parallel from adapter (adapter may internally batch)
                metrics_to_fetch = [w["metric"] for w in widgets]
                data_bundle = self.adapter.fetch_metrics_bulk(metrics_to_fetch,
                                                             window_minutes=self.config.get("data_window_minutes", 15))
                # Render each widget and calculate overlays
                for widget in widgets:
                    try:
                        self._render_widget(widget, data_bundle.get(widget["metric"], []))
                    except Exception as e:
                        # Log and continue; don't allow one widget error to stop refresh loop
                        print(f"[ERROR] rendering widget {widget.get('title')}: {e}")

                # Run periodic analyses
                try:
                    incidents = self.analyzer.correlate_recent_incidents(data_bundle)
                    capacity_report = self.planner.estimate_capacity_requirements(data_bundle)
                    self._render_summary(incidents, capacity_report)
                except Exception as e:
                    print(f"[ERROR] periodic analysis failed: {e}")

                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("🛑 Shutdown signal received; exiting dashboard loop.")

    def _render_widget(self, widget: Dict[str, Any], values: List[float]) -> None:
        # Basic textual render for CLI. Replace with actual UI rendering integration (Grafana, internal UI).
        title = widget.get("title", widget["metric"])
        latest = values[-1] if values else None
        print(f"--- {title} ---")
        print(f"Metric: {widget['metric']}")
        print(f"Latest: {latest if latest is not None else 'N/A'} (samples={len(values)})")

        # Apply alerts overlay (threshold-based)
        alerts = self.overlay.apply_overlays(widget["metric"], values, widget.get("thresholds", {}))
        if alerts:
            for a in alerts:
                print(f"🚨 {a}")

        # Provide quick micro-insights: e.g., top-3 percentiles and moving average
        if values:
            avg = round(sum(values) / len(values), 2)
            p95 = self._percentile(values, 95)
            p50 = self._percentile(values, 50)
            print(f"Avg: {avg} | p50: {p50} | p95: {p95}")

        print("")  # newline for readability

    def _render_summary(self, incidents: Dict[str, Any], capacity_report: Dict[str, Any]) -> None:
        # Summarize incident correlations and capacity recommendations to exec / ops
        print("=== Periodic Analyses ===")
        print("Incident Correlation Summary:")
        for k, v in incidents.items():
            print(f"- {k}: {v}")
        print("\nCapacity Recommendations:")
        for resource, rec in capacity_report.items():
            print(f"- {resource}: {rec}")
        print("=========================\n")

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Compute percentile (simple, deterministic)."""
        if not data:
            return 0.0
        k = (len(data) - 1) * (percentile / 100.0)
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return round(data[int(k)], 2)
        d0 = data[f] * (c - k)
        d1 = data[c] * (k - f)
        return round(d0 + d1, 2)


if __name__ == "__main__":
    dashboard = OpsDashboard("monitoring/dashboards/operations/ops_dashboard_config.json")
    dashboard.start()
