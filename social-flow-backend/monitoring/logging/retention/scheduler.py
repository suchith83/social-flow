# Scheduling system for periodic cleanup
# monitoring/logging/retention/scheduler.py
"""
Scheduler for periodic retention cleanup jobs.
"""

import threading
import time
from pathlib import Path
from .cleaner import LogCleaner
from .config import CONFIG


class RetentionScheduler:
    def __init__(self, base_path: Path):
        self.cleaner = LogCleaner(base_path)
        self.interval = CONFIG["CLEANUP"]["interval_hours"] * 3600
        self._stop = threading.Event()

    def start(self):
        """Start background scheduler thread."""
        def loop():
            while not self._stop.is_set():
                self.cleaner.clean()
                time.sleep(self.interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return thread

    def stop(self):
        self._stop.set()
