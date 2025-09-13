# Adaptive power-saving profiles
"""
Adaptive Power Profiles
Switches between high performance, balanced, and power saver modes.
"""

from .config import CONFIG


class PowerProfileManager:
    def __init__(self):
        self.mode = "BALANCED"

    def update_profile(self, battery_level: int):
        """Adjust power profile based on battery percentage."""
        if battery_level >= CONFIG.high_performance_threshold:
            self.mode = "HIGH_PERFORMANCE"
        elif battery_level >= CONFIG.balanced_threshold:
            self.mode = "BALANCED"
        elif battery_level >= CONFIG.power_saver_threshold:
            self.mode = "POWER_SAVER"
        else:
            self.mode = "CRITICAL"

    def get_current_mode(self) -> str:
        return self.mode
