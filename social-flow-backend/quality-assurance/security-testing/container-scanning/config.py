"""
Configuration file for container scanning.
Defines global constants and dynamic configurations.
"""

import os

# Registry authentication (would usually be pulled from a vault)
REGISTRY_CONFIG = {
    "default_registry": os.getenv("CONTAINER_REGISTRY", "docker.io"),
    "username": os.getenv("REGISTRY_USER", "admin"),
    "password": os.getenv("REGISTRY_PASS", "password123"),
}

# Vulnerability DB sync settings
VULN_DB_CONFIG = {
    "update_url": os.getenv("VULN_DB_URL", "https://security-tracker.example.com/db"),
    "update_interval_hours": 6,
    "local_cache": "./data/vuln_cache.json",
}

# Reporting
REPORT_CONFIG = {
    "output_dir": "./reports",
    "format": "json",  # options: json, html, pdf
    "send_to_slack": True,
    "slack_webhook": os.getenv("SLACK_WEBHOOK", None),
}
