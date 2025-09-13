# Prometheus metrics + logging
# ================================================================
# File: monitoring.py
# Purpose: Prometheus metrics and logging
# ================================================================

import logging
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger("Monitoring")


def setup_metrics(app):
    Instrumentator().instrument(app).expose(app)
    logger.info("📊 Prometheus metrics enabled at /metrics")
