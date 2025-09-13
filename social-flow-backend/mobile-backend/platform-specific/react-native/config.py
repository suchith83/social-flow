# Configuration for React Native endpoints and services
"""
Configuration for React Native platform-specific module.

Override via environment variables or secrets manager in production.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RNConfig:
    # Storage
    bundle_storage_dir: str = "/var/data/react_native/bundles"   # stores JS bundles / assets
    diffs_storage_dir: str = "/var/data/react_native/diffs"

    # Push
    fcm_server_key: str = ""
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_auth_key_path: str = ""
    fcm_batch_size: int = 1000
    apns_batch_size: int = 500

    # Diffing / OTA
    enable_bundle_diffing: bool = True
    min_diff_size_bytes: int = 1024 * 10  # 10KB
    max_delta_size_ratio: float = 0.8     # only use delta if diff < ratio * full_size

    # Device registry
    device_prune_days: int = 90

    # Signature verification
    require_signature_verification: bool = False

    # Analytics
    analytics_rate_limit_per_minute: int = 2000

    # Deep link
    allowed_deep_link_domains: Tuple[str, ...] = ("example.com",)

    # Metrics
    metrics_flush_interval_sec: int = 60

CONFIG = RNConfig()
