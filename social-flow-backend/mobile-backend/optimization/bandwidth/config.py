# Configurations, thresholds, policies
"""
Configuration for Bandwidth Optimization
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BandwidthConfig:
    # Compression settings
    compression_level: int = 6  # 1=fastest, 9=best compression
    min_compression_size: int = 1024  # bytes

    # Adaptive streaming
    default_bitrate: int = 128  # kbps
    max_bitrate: int = 2048
    min_bitrate: int = 64
    network_latency_threshold: float = 300.0  # ms

    # Request batching
    batch_window_ms: int = 200
    max_batch_size: int = 50

    # Predictive caching
    cache_size_limit_mb: int = 50
    prediction_confidence_threshold: float = 0.7

    # Network throttling
    max_bandwidth_per_user_kbps: int = 500
    throttle_check_interval_sec: int = 5

    # Metrics
    metrics_flush_interval_sec: int = 60


CONFIG = BandwidthConfig()
