# Network-aware optimizations for battery
"""
Adaptive Network Optimizer
Reduces battery drain from networking by preferring WiFi,
throttling cellular, and deferring sync tasks on low battery.
"""

from .config import CONFIG


class AdaptiveNetworkOptimizer:
    def __init__(self):
        self.last_used_network = "wifi"

    def choose_network(self, battery_level: int, networks_available: dict) -> str:
        """
        Decide whether to use WiFi or cellular.
        networks_available = {"wifi": True/False, "cellular": True/False}
        """
        if CONFIG.wifi_preference and networks_available.get("wifi", False):
            self.last_used_network = "wifi"
        elif (
            networks_available.get("cellular", False)
            and battery_level > CONFIG.cellular_throttle_threshold
        ):
            self.last_used_network = "cellular"
        else:
            self.last_used_network = "offline"
        return self.last_used_network
