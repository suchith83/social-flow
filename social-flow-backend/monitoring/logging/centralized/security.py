# Redaction, masking, and RBAC enforcement
# monitoring/logging/centralized/security.py
"""
Security layer for centralized logging.
Includes field masking and RBAC enforcement.
"""

from .config import CONFIG


def mask_sensitive(log: dict) -> dict:
    """Mask sensitive fields defined in config."""
    masked = log.copy()
    for field in CONFIG["SECURITY"]["mask_fields"]:
        if field in masked:
            masked[field] = "***REDACTED***"
    return masked


class RBAC:
    """Simple RBAC enforcement for log access."""

    def __init__(self):
        self.roles = {
            "admin": {"read": True, "write": True},
            "analyst": {"read": True, "write": False},
            "guest": {"read": False, "write": False}
        }

    def can_read(self, role: str) -> bool:
        return self.roles.get(role, {}).get("read", False)

    def can_write(self, role: str) -> bool:
        return self.roles.get(role, {}).get("write", False)
