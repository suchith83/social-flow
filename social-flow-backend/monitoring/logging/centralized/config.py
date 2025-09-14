# Configurations for centralized logging
# monitoring/logging/centralized/config.py
"""
Centralized logging configuration.
Defines backend settings, batching thresholds, and security policies.
"""

from pathlib import Path

CONFIG = {
    "COLLECTOR": {
        "batch_size": 100,
        "flush_interval_sec": 5,
        "supported_sources": ["file", "api", "stream"]
    },
    "STORAGE": {
        "backend": "inmemory",  # could be "elasticsearch", "postgres"
        "retention_days": 30,
        "data_dir": Path("centralized_storage")
    },
    "INDEXER": {
        "enable_full_text": True,
        "max_index_size": 10_000_000
    },
    "SECURITY": {
        "mask_fields": ["password", "ssn", "credit_card"],
        "rbac_enabled": True
    }
}
