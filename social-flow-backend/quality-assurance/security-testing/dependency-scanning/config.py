"""
Configuration for Dependency Scanning
"""

import os

# Which ecosystems to support
SUPPORTED_ECOSYSTEMS = ["python", "node", "java"]

# Vulnerability DB sources
VULN_DB_CONFIG = {
    "osv_api": "https://api.osv.dev/v1/query",
    "cache_file": "./data/dependency_vuln_cache.json",
    "update_interval_hours": 12
}

# Reporting
REPORT_CONFIG = {
    "output_dir": "./dependency-reports",
    "format": "json",  # options: json, html, pdf
    "slack_notify": bool(os.getenv("SLACK_WEBHOOK")),
    "slack_webhook": os.getenv("SLACK_WEBHOOK")
}
