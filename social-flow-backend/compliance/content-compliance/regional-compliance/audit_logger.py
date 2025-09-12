"""
audit_logger.py

Regional audit logger: writes region-aware logs to filesystem for later ingestion by SIEM.
Provides structured events (JSON).
"""

import json
from datetime import datetime
from pathlib import Path
import logging

LOG_BASE = Path("logs/compliance/regional")
LOG_BASE.mkdir(parents=True, exist_ok=True)

# Simple logger wrapper
class RegionalAuditLogger:
    @staticmethod
    def _now_iso():
        return datetime.utcnow().isoformat()

    @staticmethod
    def log_event(event_type: str, actor: str, details: dict, jurisdiction: str = "global"):
        filename = LOG_BASE / f"{jurisdiction}_audit.log"
        record = {
            "timestamp": RegionalAuditLogger._now_iso(),
            "event": event_type,
            "actor": actor,
            "jurisdiction": jurisdiction,
            "details": details
        }
        with open(filename, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        # Also emit to standard logging for immediate debug
        logging.getLogger(__name__).info(f"[{jurisdiction}] {event_type} by {actor}: {details}")
