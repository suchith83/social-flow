# Configurations & thresholds
"""
Performance Optimization Configurations
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceConfig:
    # Task scheduling
    max_priority: int = 10
    default_priority: int = 5

    # Thread pool
    min_threads: int = 4
    max_threads: int = 32

    # Caching
    memory_cache_size_mb: int = 100
    persistent_cache_ttl_sec: int = 300

    # Load balancer
    max_nodes: int = 10
    rebalance_interval_sec: int = 60

    # Query optimizer
    query_cache_ttl_sec: int = 120
    query_batch_size: int = 50

    # Metrics
    metrics_flush_interval_sec: int = 60


CONFIG = PerformanceConfig()
