# Define and evaluate Service Level Objectives
"""
slo.py
Helpers for computing SLOs and evaluating compliance periodically.
Example SLOs:
 - Availability: percentage of successful inferences over total in window
 - Latency: p95 latency < threshold

This file includes utilities to:
 - compute SLOs from Prometheus via HTTP API
 - evaluate compliance and produce alerts (local policy)
"""

import requests
from typing import Dict
from utils import setup_logger
import time

logger = setup_logger("SLO")


class SLOClient:
    def __init__(self, prometheus_base: str, timeout: int = 10):
        self.base = prometheus_base.rstrip("/")
        self.timeout = timeout

    def query(self, promql: str):
        url = f"{self.base}/api/v1/query"
        r = requests.get(url, params={"query": promql}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data["status"] != "success":
            raise RuntimeError("Prometheus query failed")
        return data["data"]["result"]

    def availability(self, job: str, window: str = "5m") -> float:
        # fraction of success requests / total
        promql = f'(sum(increase(model_inference_total{{model="{job}",status="success"}}[{window}])) by (model) / (sum(increase(model_inference_total{{model="{job}"}}[{window}])) by (model) + 0.0))'
        res = self.query(promql)
        if not res:
            return 0.0
        return float(res[0]["value"][1])

    def latency_percentile(self, job: str, quantile: float = 0.95, window: str = "5m") -> float:
        # Example using histogram_quantile with buckets
        promql = f'histogram_quantile({quantile}, sum(rate(model_inference_latency_seconds_bucket{{model="{job}"}}[{window}])) by (le))'
        res = self.query(promql)
        if not res:
            return 0.0
        return float(res[0]["value"][1])


def evaluate_slo(prom_base: str, job: str, availability_threshold: float = 0.99, latency_threshold: float = 1.0):
    cli = SLOClient(prom_base)
    avail = cli.availability(job)
    lat = cli.latency_percentile(job, 0.95)
    logger.info(f"SLO evaluation for {job}: availability={avail:.4f}, p95_latency={lat:.3f}s")
    violations = []
    if avail < availability_threshold:
        violations.append(("availability", avail))
    if lat > latency_threshold:
        violations.append(("latency_p95", lat))
    return violations
