"""
enforcement.py

Contains enforcement logic to handle violations, warnings,
and escalation paths for age-restriction compliance.
"""

from typing import Optional
from datetime import datetime

from .exceptions import AgeRestrictionViolation
from .audit_logger import AuditLogger


class EnforcementAction:
    """Defines enforcement actions when violations occur."""

    @staticmethod
    def block_access(user_id: str, reason: str) -> None:
        """Block user access to content."""
        AuditLogger.log_event("BLOCK_ACCESS", user_id, details={"reason": reason})

    @staticmethod
    def warn_user(user_id: str, reason: str) -> None:
        """Send warning notification."""
        AuditLogger.log_event("WARN_USER", user_id, details={"reason": reason})

    @staticmethod
    def escalate(user_id: str, reason: str, admin_id: Optional[str] = None) -> None:
        """Escalate violation to compliance admins."""
        AuditLogger.log_event(
            "ESCALATE", user_id, details={"reason": reason, "admin": admin_id}
        )


class EnforcementEngine:
    """Engine to handle enforcement workflow."""

    @staticmethod
    def handle_violation(user_id: str, violation: AgeRestrictionViolation) -> None:
        """Process a violation with multiple levels of enforcement."""
        reason = violation.message
        # Step 1: Block access
        EnforcementAction.block_access(user_id, reason)

        # Step 2: Warn user
        EnforcementAction.warn_user(user_id, reason)

        # Step 3: Escalate if repeated or severe
        if "restricted" in reason.lower() or "adult" in reason.lower():
            EnforcementAction.escalate(user_id, reason)
