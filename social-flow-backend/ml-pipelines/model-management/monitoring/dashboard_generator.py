# Build or update Grafana dashboards
"""
dashboard_generator.py
Generate a Grafana v9+ dashboard JSON for model observability:
- Model health panel
- Inference rate & error rate
- Latency histogram / percentiles
- Input size distribution
- Model drift & data quality panels (placeholders tied to metrics)
"""

import json
from utils import write_file, setup_logger

logger = setup_logger("DashboardGenerator")


def generate_model_dashboard(model_name: str, title: str = None):
    """
    Returns a Grafana dashboard as Python dict (JSON-ready).
    This is a simplified dashboard skeleton. For full dashboards, use grafanalib or panel builders.
    """
    title = title or f"{model_name} - Model Observability"
    uid = f"{model_name}-observability"
    # panels minimal examples: In real dashboards you'd craft targets and panel configs carefully.
    dashboard = {
        "id": None,
        "uid": uid,
        "title": title,
        "timezone": "browser",
        "panels": [
            {
                "type": "stat",
                "title": "Inference Rate (r/s)",
                "id": 1,
                "targets": [
                    {"expr": f"sum(rate(model_inference_total{{model=\"{model_name}\",status=\"success\"}}[1m]))"}
                ],
                "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
            },
            {
                "type": "graph",
                "title": "Inference Latency (p50/p95/p99)",
                "id": 2,
                "targets": [
                    {"expr": f'histogram_quantile(0.50, sum(rate(model_inference_latency_seconds_bucket{{model="{model_name}"}}[5m])) by (le))', "legendFormat": "p50"},
                    {"expr": f'histogram_quantile(0.95, sum(rate(model_inference_latency_seconds_bucket{{model="{model_name}"}}[5m])) by (le))', "legendFormat": "p95"},
                    {"expr": f'histogram_quantile(0.99, sum(rate(model_inference_latency_seconds_bucket{{model="{model_name}"}}[5m])) by (le))', "legendFormat": "p99"},
                ],
                "gridPos": {"x": 6, "y": 0, "w": 12, "h": 8},
            },
            {
                "type": "stat",
                "title": "Errors (last 5m)",
                "id": 3,
                "targets": [{"expr": f'sum(rate(model_inference_total{{model="{model_name}",status="failure"}}[5m]))'}],
                "gridPos": {"x": 0, "y": 4, "w": 6, "h": 4},
            },
            {
                "type": "table",
                "title": "Drift Score (placeholder)",
                "id": 4,
                "targets": [{"expr": f"model_drift_score{{model=\"{model_name}\"}}"}],
                "gridPos": {"x": 0, "y": 8, "w": 12, "h": 6},
            }
        ],
        "schemaVersion": 30,
        "version": 1,
        "refresh": "10s",
    }
    return dashboard


def save_dashboard(dashboard_json: dict, out_path: str = "manifests/grafana_dashboard.json"):
    write_file(out_path, json.dumps(dashboard_json, indent=2))
    logger.info(f"Saved dashboard to {out_path}")
    return out_path
