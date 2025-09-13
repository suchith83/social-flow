# Prevents unnecessary device wake-ups
"""
Wake Lock Manager
Prevents unnecessary device wake-ups that drain battery.
"""

import time
from .config import CONFIG


class WakeLockManager:
    def __init__(self):
        self.active_wakes = {}

    def acquire(self, tag: str) -> bool:
        """Acquire a wake lock for a task."""
        if tag not in self.active_wakes:
            self.active_wakes[tag] = time.time()
            return True
        return False

    def release(self, tag: str) -> None:
        """Release a wake lock."""
        if tag in self.active_wakes:
            del self.active_wakes[tag]

    def enforce_limits(self):
        """Release wake locks that exceed duration limits."""
        now = time.time()
        for tag, start in list(self.active_wakes.items()):
            if now - start > CONFIG.max_wake_duration_sec:
                del self.active_wakes[tag]
