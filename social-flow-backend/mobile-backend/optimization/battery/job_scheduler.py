# Smart background job scheduling
"""
Smart Job Scheduler
Defers non-critical background tasks until favorable conditions are met.
"""

import time
from typing import Callable, List, Dict
from .config import CONFIG


class SmartJobScheduler:
    def __init__(self):
        self.queue: List[Dict] = []

    def schedule(self, job_fn: Callable, battery_level: int, is_charging: bool) -> bool:
        """Schedule a job only if battery allows."""
        if battery_level >= CONFIG.min_battery_percent_for_heavy_jobs or is_charging:
            job_fn()
            return True
        else:
            self.queue.append({"fn": job_fn, "time": time.time()})
            return False

    def flush(self, battery_level: int, is_charging: bool):
        """Run queued jobs when conditions improve."""
        runnable = []
        if battery_level >= CONFIG.min_battery_percent_for_heavy_jobs or is_charging:
            for job in self.queue:
                runnable.append(job["fn"])
            self.queue.clear()
            for fn in runnable:
                fn()
