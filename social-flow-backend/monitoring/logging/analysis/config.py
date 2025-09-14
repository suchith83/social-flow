# Configuration for thresholds, models, and pipelines
# monitoring/logging/analysis/config.py
"""
Configuration module for log analysis.
Defines thresholds, models, and system-wide settings for anomaly detection,
trend analysis, correlation, and reporting.
"""

from pathlib import Path

CONFIG = {
    "LOG_SOURCES": ["syslog", "json", "custom"],
    "PARSER": {
        "timestamp_formats": [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%b %d %H:%M:%S"
        ]
    },
    "ANOMALY_DETECTOR": {
        "zscore_threshold": 3.0,
        "isolation_forest": {
            "n_estimators": 200,
            "contamination": 0.01,
            "random_state": 42
        }
    },
    "TREND_ANALYSIS": {
        "aggregation_window": "5min",
        "rolling_window": 12
    },
    "CORRELATION": {
        "time_tolerance_sec": 30,
        "causal_score_threshold": 0.7
    },
    "REPORTS": {
        "output_dir": Path("reports"),
        "formats": ["json", "html"]
    }
}
