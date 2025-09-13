# Prometheus/Alerting hooks & service monitor manifests
# ================================================================
# File: monitor_integration.py
# Purpose: Generate PrometheusServiceMonitor manifest + Alertmanager rule skeleton
#          and expose endpoints to register scraping annotations.
# ================================================================

from utils import write_file, logger

SERVICE_MONITOR_TPL = """
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {name}-servicemonitor
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: {name}
  endpoints:
    - port: http
      path: /metrics
      interval: {interval}s
"""

ALERT_RULE_TPL = """
groups:
- name: {name}-alerts
  rules:
  - alert: {name}HighErrorRate
    expr: job:request_errors:rate5m{{job="{name}"}} > {threshold}
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate for {name}"
      description: "Error rate for job {name} exceeded {threshold}"
"""

def generate_service_monitor(name: str, out_dir: str = "manifests", interval: int = 15):
    path = f"{out_dir}/{name}_servicemonitor.yaml"
    content = SERVICE_MONITOR_TPL.format(name=name, interval=interval)
    write_file(path, content)
    logger.info(f"Generated ServiceMonitor at {path}")
    return path

def generate_alert_rule(name: str, out_dir: str = "manifests", threshold: float = 0.05):
    path = f"{out_dir}/{name}_alert_rule.yaml"
    content = ALERT_RULE_TPL.format(name=name, threshold=threshold)
    write_file(path, content)
    logger.info(f"Generated alert rule at {path}")
    return path
