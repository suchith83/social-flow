"""
audit_logger.py

Robust audit logging tuned for copyright workflows.
Writes JSON lines to a rotating log file and also prints structured messages to console for dev.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("logs/compliance/copyright")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "copyright_audit.log"

logger = logging.getLogger("copyright_audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class AuditLogger:
    @staticmethod
    def log_event(event_type: str, actor: str, details: dict) -> None:
        record = {
            "timestamp": _now_iso(),
            "event": event_type,
            "actor": actor,
            "details": details,
        }
        logger.info(json.dumps(record))
        # Also print for immediate feedback when running locally
        print(f"[AUDIT] {event_type} by {actor}: {json.dumps(details)}")
