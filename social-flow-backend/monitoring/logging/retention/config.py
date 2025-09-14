# Retention-related configurations
# monitoring/logging/retention/config.py
"""
Retention configuration module.
Defines storage tiers, time thresholds, and archiving preferences.
"""

from pathlib import Path

CONFIG = {
    "DEFAULT_RETENTION_DAYS": 30,
    "TIERED_STORAGE": {
        "hot": 7,      # logs < 7 days
        "warm": 30,    # logs between 7–30 days
        "cold": 90,    # logs between 30–90 days
        "archive": 365 # logs older than 90 days compressed
    },
    "ARCHIVE": {
        "enabled": True,
        "path": Path("archives"),
        "compression": "gzip"  # gzip | zip
    },
    "CLEANUP": {
        "enabled": True,
        "interval_hours": 24
    }
}
