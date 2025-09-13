# Configuration for Flutter endpoints and services
"""
Configuration for Flutter platform-specific backend module.

Override via environment variables or secrets manager in production.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlutterConfig:
    # Artifact storage
    bundle_storage_dir: str = "/var/data/flutter/bundles"  # stores .app/.aab/.ipa or flutter bundles
    diffs_storage_dir: str = "/var/data/flutter/diffs"

    # Push
    fcm_server_key: str = ""
    apns_key_path: str = ""  # path to APNs auth key for iOS push (p8)
    fcm_batch_size: int = 1000

    # Diffing
    enable_bundle_diffing: bool = True
    min_diff_size_bytes: int = 1024 * 20  # only attempt diffs if >20KB

    # Device registry & pruning
    device_prune_days: int = 90

    # Analytics
    analytics_rate_limit_per_minute: int = 1500

    # Signature verification
    require_signature_verification: bool = True

    # Metrics
    metrics_flush_interval_sec: int = 60


CONFIG = FlutterConfig()
