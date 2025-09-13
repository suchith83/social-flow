# Configuration for iOS endpoints and services
"""
Configuration for iOS platform-specific backend module.

Override via environment variables or a secrets manager in production.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class iOSConfig:
    # Storage
    ipa_storage_dir: str = "/var/data/ios/ipas"
    diffs_storage_dir: str = "/var/data/ios/diffs"

    # APNs (push)
    apns_key_id: str = ""          # Key ID in Apple Developer portal
    apns_team_id: str = ""         # Team ID
    apns_auth_key_path: str = ""   # Path to .p8 file for JWT
    apns_topic: str = ""           # Bundle id for push topic
    apns_batch_size: int = 1000

    # Diffing
    enable_ipa_diffing: bool = True
    min_diff_size_bytes: int = 1024 * 50

    # Device registry & pruning
    device_prune_days: int = 90

    # Signature verification
    require_signature_verification: bool = True

    # Analytics
    analytics_rate_limit_per_minute: int = 1200

    # Deep links / universal links
    allowed_universal_link_domains: tuple = ("example.com",)

    # Metrics
    metrics_flush_interval_sec: int = 60


CONFIG = iOSConfig()
