"""
Configuration for Dynamic Analysis (DAST)
"""

import os

# Targets
SCAN_CONFIG = {
    "base_url": os.getenv("TARGET_URL", "http://localhost:8080"),
    "timeout": 10,
    "max_depth": 3,
    "include_subdomains": True,
}

# Attack payloads
PAYLOADS = {
    "xss": ["<script>alert(1)</script>", "\"'><img src=x onerror=alert(1)>"],
    "sqli": ["' OR '1'='1", "\" OR \"1\"=\"1", "'; DROP TABLE users; --"],
    "ssrf": ["http://127.0.0.1:22", "file:///etc/passwd"],
}

# Reporting
REPORT_CONFIG = {
    "output_dir": "./dast-reports",
    "format": "json",  # options: json, html
}
