# Parses raw log formats (JSON, syslog, custom)
# monitoring/logging/analysis/log_parser.py
"""
Advanced log parser.
Handles multiple formats: JSON logs, syslog, and custom structured text logs.
"""

import re
import json
from .utils import parse_timestamp
from .config import CONFIG


class LogParser:
    def __init__(self):
        self.timestamp_formats = CONFIG["PARSER"]["timestamp_formats"]

    def parse(self, raw_log: str) -> dict:
        """Detect log type and parse accordingly."""
        if raw_log.strip().startswith("{"):
            return self._parse_json(raw_log)
        elif re.match(r"^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", raw_log):
            return self._parse_syslog(raw_log)
        else:
            return self._parse_custom(raw_log)

    def _parse_json(self, raw_log: str) -> dict:
        try:
            log = json.loads(raw_log)
            log["timestamp"] = parse_timestamp(
                log.get("timestamp", ""), self.timestamp_formats
            )
            return log
        except Exception:
            return {"raw": raw_log, "error": "Invalid JSON"}

    def _parse_syslog(self, raw_log: str) -> dict:
        match = re.match(r"^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}) (\S+) (.+)", raw_log)
        if not match:
            return {"raw": raw_log, "error": "Invalid syslog format"}
        ts_str, host, message = match.groups()
        return {
            "timestamp": parse_timestamp(ts_str, self.timestamp_formats),
            "host": host,
            "message": message,
        }

    def _parse_custom(self, raw_log: str) -> dict:
        parts = raw_log.split("|")
        if len(parts) < 3:
            return {"raw": raw_log, "error": "Invalid custom log format"}
        return {
            "timestamp": parse_timestamp(parts[0], self.timestamp_formats),
            "level": parts[1],
            "message": "|".join(parts[2:]),
        }
