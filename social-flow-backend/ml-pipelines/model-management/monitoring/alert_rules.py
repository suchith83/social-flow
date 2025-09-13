# Alerting rules definitions and loader
"""
alert_rules.py
Generate Prometheus Alertmanager rules YAML and simple best-practice SLO-based alerts.
Exports:
 - generate_alert_rule(name, expr, for_minutes, severity, summary, description)
 - generate_error_rate_alert(name, job_label, threshold, for_minutes)
 - generate_latency_alert(name, job_label, metric_name, threshold_seconds, percentile, for_minutes)
"""

import yaml
from utils import write_file, setup_logger

logger = setup_logger("AlertRules")


def _format_rule(rule: dict) -> dict:
    return rule


def generate_alert_rule_file(rules: list, out_path: str = "manifests/alert_rules.yaml"):
    """
    rules: list of rule dicts with keys: name, expr, for, labels, annotations
    """
    file_contents = {"groups": [{"name": "ml-model-alerts", "rules": rules}]}
    write_file(out_path, yaml.safe_dump(file_contents))
    logger.info(f"Wrote alert rules to {out_path}")
    return out_path


def error_rate_rule(job_label: str, threshold: float = 0.05, window: str = "5m", name: str = None):
    """
    Generate rule to alert when error rate (non-2xx responses) for a job exceeds threshold.
    Assumes an exported metric `http_requests_total{job="<job_label>", status=~"5.."}`
    or using default pattern: increase(errors[5m]) / increase(requests[5m]) > threshold
    """
    rule_name = name or f"{job_label}_high_error_rate"
    expr = f'(sum(rate(http_requests_total{{job="{job_label}",status=~"5.."}}[{window}])) by (job) / (sum(rate(http_requests_total{{job="{job_label}"}}[{window}])) by (job) + 1e-9)) > {threshold}'
    return {
        "alert": rule_name,
        "expr": expr,
        "for": f"{window}",
        "labels": {"severity": "critical"},
        "annotations": {
            "summary": f"High error rate for {job_label}",
            "description": f"Error rate for job {job_label} is above {threshold} over {window}"
        }
    }


def latency_rule(job_label: str, metric_name: str = "model_inference_latency_seconds", quantile: float = 0.99, threshold_seconds: float = 1.0, window: str = "5m", name: str = None):
    """
    Alert if high percentile latency for a job exceeds threshold.
    Example expr uses histogram_quantile if using summary/histogram metrics. Here we craft a generic expr.
    """
    rule_name = name or f"{job_label}_high_latency"
    # This expr assumes you have histogram buckets; adjust to your metric form
    expr = f'histogram_quantile({quantile}, sum(rate({metric_name}_bucket{{job="{job_label}"}}[{window}])) by (le)) > {threshold_seconds}'
    return {
        "alert": rule_name,
        "expr": expr,
        "for": f"{window}",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": f"High latency for {job_label}",
            "description": f"{quantile*100:.0f}th percentile latency > {threshold_seconds}s for {job_label}"
        }
    }
