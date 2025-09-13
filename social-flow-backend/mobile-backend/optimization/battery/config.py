# Configurations and thresholds
"""
Battery Optimization Configurations
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    # Job scheduler
    min_battery_percent_for_heavy_jobs: int = 40
    background_job_window_sec: int = 300

    # Power profiles
    high_performance_threshold: int = 80
    balanced_threshold: int = 40
    power_saver_threshold: int = 15

    # Resource throttling
    cpu_throttle_low: float = 0.3
    cpu_throttle_high: float = 0.7
    gpu_throttle: float = 0.5

    # Wake lock
    max_wake_duration_sec: int = 60

    # Adaptive network
    wifi_preference: bool = True
    cellular_throttle_threshold: int = 20  # Battery % below which cellular use is restricted

    # Metrics
    metrics_flush_interval_sec: int = 60


CONFIG = BatteryConfig()
