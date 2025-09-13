# Configuration for Android endpoints and services
"""
Configuration for Android platform-specific module.

In production, override values via environment variables or a secrets manager.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AndroidConfig:
    # App & artifact storage (local directories for demo; replace with S3/GCS)
    apk_storage_dir: str = "/var/data/android/apks"
    diffs_storage_dir: str = "/var/data/android/diffs"

    # FCM / Push
    fcm_server_key: str = ""  # set from env / secret
    fcm_api_url: str = "https://fcm.googleapis.com/fcm/send"
    fcm_batch_size: int = 1000

    # APK diffing
    enable_apk_diffing: bool = True
    min_diff_size_bytes: int = 1024 * 50  # only attempt diffs if >50KB

    # Device registry
    device_id_hash_salt: str = "change-me-in-prod"
    device_prune_days: int = 90

    # Rate limiting (ingestion endpoints)
    analytics_rate_limit_per_minute: int = 1200
    push_rate_limit_per_minute: int = 500

    # Metrics
    metrics_flush_interval_sec: int = 60

    # Security
    allowed_deep_link_domains: tuple = ("example.com",)

    # Signature verification
    require_signature_verification: bool = True


CONFIG = AndroidConfig()
