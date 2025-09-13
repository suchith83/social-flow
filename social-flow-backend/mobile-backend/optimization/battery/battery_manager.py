# Orchestrates optimization strategies
"""
Battery Manager
Coordinates all battery optimization modules.
"""

from .job_scheduler import SmartJobScheduler
from .power_profiles import PowerProfileManager
from .resource_throttling import ResourceThrottler
from .wake_lock_manager import WakeLockManager
from .adaptive_network import AdaptiveNetworkOptimizer
from .metrics import BatteryMetrics


class BatteryManager:
    def __init__(self):
        self.scheduler = SmartJobScheduler()
        self.profiles = PowerProfileManager()
        self.throttler = ResourceThrottler()
        self.wakelocks = WakeLockManager()
        self.network = AdaptiveNetworkOptimizer()
        self.metrics = BatteryMetrics()

    def optimize(self, battery_level: int, is_charging: bool, networks_available: dict):
        """Main loop for optimizing system behavior."""
        self.profiles.update_profile(battery_level)
        mode = self.profiles.get_current_mode()

        # Throttle resources
        self.throttler.adjust_resources(battery_level, mode)

        # Adaptive networking
        chosen_network = self.network.choose_network(battery_level, networks_available)

        # Metrics logging
        self.metrics.log("battery_level", battery_level)
        self.metrics.log("cpu_limit", self.throttler.cpu_limit)
        self.metrics.log("gpu_limit", self.throttler.gpu_limit)

        return {"mode": mode, "network": chosen_network}

    def run_task(self, task_name: str, battery_level: int):
        """Execute a task under resource throttling."""
        cost = self.throttler.execute_task(task_name)
        self.metrics.log("task_cost", cost)
        return cost

    def flush_metrics(self) -> dict:
        return self.metrics.flush()
