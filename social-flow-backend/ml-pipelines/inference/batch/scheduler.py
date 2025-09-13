# Batch scheduling (cron/Airflow integration)
# ================================================================
# File: scheduler.py
# Purpose: Batch job scheduling (cron / airflow)
# ================================================================

import logging
import schedule
import time

logger = logging.getLogger("Scheduler")


class Scheduler:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.interval = config.get("interval", 60)

    def start(self, job_fn):
        logger.info(f"Scheduler started: every {self.interval} seconds")
        schedule.every(self.interval).seconds.do(job_fn)
        while True:
            schedule.run_pending()
            time.sleep(1)
