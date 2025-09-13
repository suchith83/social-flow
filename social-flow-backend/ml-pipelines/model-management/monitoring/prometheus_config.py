# Prometheus configuration management
"""
prometheus_config.py
Generate a minimal Prometheus scrape config for model services and exporters.
 - generate_scrape_config(job_name, targets, metrics_path, scrape_interval)
 - generate_full_config(global_config, scrape_jobs, out_path)
"""

import yaml
from utils import write_file, setup_logger

logger = setup_logger("PromConfig")


def generate_scrape_job(job_name: str, targets: list, metrics_path: str = "/metrics", scrape_interval: str = "15s"):
    return {
        "job_name": job_name,
        "metrics_path": metrics_path,
        "static_configs": [{"targets": targets}]
    }


def generate_prometheus_config(global_config: dict = None, scrape_jobs: list = None, out_path: str = "manifests/prometheus.yml"):
    cfg = {
        "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
        "scrape_configs": scrape_jobs or []
    }
    if global_config:
        cfg["global"].update(global_config)
    write_file(out_path, yaml.safe_dump(cfg))
    logger.info(f"Generated prometheus config at {out_path}")
    return out_path
