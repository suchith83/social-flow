# Logging, metrics, and error tracking
# ================================================================
# File: monitoring.py
# Purpose: Logging, metrics, and result storage
# ================================================================

import logging
import json
import time
from pathlib import Path

logger = logging.getLogger("Monitoring")


class Monitoring:
    def __init__(self, config: dict):
        self.results_path = Path(config.get("results_path", "outputs/batch_results"))
        self.results_path.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, job, duration, records):
        logger.info(f"[{job}] duration={duration:.2f}s records={records}")

    def save_results(self, df):
        ts = int(time.time())
        path = self.results_path / f"batch_results_{ts}.json"
        df.to_json(path, orient="records")
        logger.info(f"Results saved to {path}")
