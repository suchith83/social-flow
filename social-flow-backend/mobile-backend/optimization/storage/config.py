# Configuration for storage tiers, deduplication, compression
"""
Configurations for storage optimization module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageConfig:
    # Tier sizes (bytes)
    hot_tier_max_bytes: int = 50 * 1024 * 1024  # 50 MB (per object threshold)
    warm_tier_max_bytes: int = 500 * 1024 * 1024  # 500 MB

    # Retention and lifecycle
    hot_to_warm_days: int = 1
    warm_to_cold_days: int = 7
    cold_retention_days: int = 365

    # Deduplication
    enable_dedup: bool = True
    dedup_min_size: int = 1024  # bytes (skip tiny objects)

    # Compression
    default_algorithm: str = "brotli"  # choices: zlib, lzma, brotli
    compression_level: int = 6

    # Async writer
    write_queue_max: int = 1000
    write_batch_size: int = 32
    write_flush_interval_sec: int = 2

    # Garbage collection
    gc_run_interval_sec: int = 60 * 60  # hourly
    gc_delete_batch: int = 100

    # Metrics
    metrics_flush_interval_sec: int = 60


CONFIG = StorageConfig()
