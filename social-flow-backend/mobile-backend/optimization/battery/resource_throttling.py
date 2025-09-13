# Throttles CPU/GPU-intensive tasks
"""
Resource Throttler
Reduces CPU/GPU load for battery savings.
"""

import random
from .config import CONFIG


class ResourceThrottler:
    def __init__(self):
        self.cpu_limit = 1.0
        self.gpu_limit = 1.0

    def adjust_resources(self, battery_level: int, mode: str):
        """Throttle CPU/GPU usage based on power mode."""
        if mode == "HIGH_PERFORMANCE":
            self.cpu_limit = 1.0
            self.gpu_limit = 1.0
        elif mode == "BALANCED":
            self.cpu_limit = CONFIG.cpu_throttle_high
            self.gpu_limit = CONFIG.gpu_throttle
        elif mode == "POWER_SAVER":
            self.cpu_limit = CONFIG.cpu_throttle_low
            self.gpu_limit = CONFIG.gpu_throttle * 0.7
        else:  # CRITICAL
            self.cpu_limit = 0.1
            self.gpu_limit = 0.1

    def execute_task(self, task_name: str) -> float:
        """Simulate task execution under throttling."""
        workload = random.uniform(0.1, 1.0)
        return workload * self.cpu_limit
