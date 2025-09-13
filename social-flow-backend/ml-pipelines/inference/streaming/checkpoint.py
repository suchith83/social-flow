# State & checkpointing for exactly-once
# ================================================================
# File: monitoring.py
# Purpose: Metrics and logging for streaming inference
# ================================================================

import logging
import prometheus_client

logger = logging.getLogger("Monitoring")


class Monitoring:
    def __init__(self, config: dict):
        self.inference_counter = prometheus_client.Counter(
            "inference_requests_total", "Total inference requests", ["status"]
        )

    def log_inference(self, success: bool):
        status = "success" if success else "failure"
        self.inference_counter.labels(status=status).inc()
        logger.info(f"📊 Inference {status}")
