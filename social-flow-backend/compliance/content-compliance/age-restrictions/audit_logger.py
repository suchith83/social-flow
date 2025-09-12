"""
audit_logger.py

Centralized audit logger for all age restriction compliance events.
Ensures traceability and regulatory compliance.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

# Setup rotating log file
LOG_DIR = Path("logs/compliance/age_restrictions")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "audit.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class AuditLogger:
    """Audit logging utility."""

    @staticmethod
    def log_event(event_type: str, user_id: str, details: dict) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "user_id": user_id,
            "details": details,
        }
        logging.info(json.dumps(record))
