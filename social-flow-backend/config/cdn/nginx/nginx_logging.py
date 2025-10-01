"""
# Logging management & log parsing
"""
# config/cdn/nginx/nginx_logging.py
"""
Nginx Logging
-------------
Configure access & error logs, and log parsing.
"""

import re
from typing import List, Dict


class NginxLogging:
    @staticmethod
    def log_format() -> str:
        """Custom log format for better analytics."""
        return """
log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                '$status $body_bytes_sent "$http_referer" '
                '"$http_user_agent" "$http_x_forwarded_for"';
access_log /var/log/nginx/access.log main;
error_log /var/log/nginx/error.log warn;
"""

    @staticmethod
    def parse_access_log(filepath: str) -> List[Dict[str, str]]:
        """Simple parser for Nginx access log."""
        logs = []
        regex = re.compile(
            r'(?P<ip>\S+) - \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d{3}) (?P<size>\d+)'
        )
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = regex.match(line)
                if match:
                    logs.append(match.groupdict())
        return logs
