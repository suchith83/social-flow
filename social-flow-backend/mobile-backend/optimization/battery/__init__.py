# Package initializer for battery module
"""
Battery Optimization Package

Provides advanced strategies for minimizing battery usage in mobile backends.
Includes adaptive job scheduling, wake lock prevention, network-aware optimizations,
and resource throttling. All modules integrate with BatteryManager.
"""

from .battery_manager import BatteryManager
from .job_scheduler import SmartJobScheduler
from .power_profiles import PowerProfileManager
from .resource_throttling import ResourceThrottler
from .wake_lock_manager import WakeLockManager
from .adaptive_network import AdaptiveNetworkOptimizer
from .metrics import BatteryMetrics

__all__ = [
    "BatteryManager",
    "SmartJobScheduler",
    "PowerProfileManager",
    "ResourceThrottler",
    "WakeLockManager",
    "AdaptiveNetworkOptimizer",
    "BatteryMetrics",
]
