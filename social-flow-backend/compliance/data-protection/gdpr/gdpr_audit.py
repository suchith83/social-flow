"""
# GDPR audit logs and reporting utilities
"""
"""
GDPR Audit Logging
------------------
"""

import logging
from .gdpr_exceptions import GDPRAuditError

class GDPRAuditLogger:
    """Audit logger for GDPR compliance events."""

    def __init__(self, log_file="gdpr_audit.log"):
        self.logger = logging.getLogger("GDPRAudit")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_event(self, event_type: str, user_id: str, details: str):
        try:
            entry = f"EVENT={event_type} | USER={user_id} | DETAILS={details}"
            self.logger.info(entry)
        except Exception as e:
            raise GDPRAuditError(f"Failed to log event: {e}")

    def log_request(self, request_id: str, user_id: str, request_type: str):
        self.log_event("REQUEST", user_id, f"RequestID={request_id} Type={request_type}")

    def log_consent(self, user_id: str, status: str):
        self.log_event("CONSENT", user_id, f"Consent={status}")

    def log_portability(self, user_id: str, file_path: str):
        self.log_event("PORTABILITY", user_id, f"DataExportedTo={file_path}")
