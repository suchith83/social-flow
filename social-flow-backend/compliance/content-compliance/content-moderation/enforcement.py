"""
enforcement.py

Handles enforcement actions for moderation violations:
warnings, takedowns, suspensions, escalations.
"""

from typing import Optional
from .audit_logger import AuditLogger
from .exceptions import ModerationViolation


class EnforcementAction:
    """Defines moderation enforcement actions."""

    @staticmethod
    def warn_user(user_id: str, reason: str) -> None:
        AuditLogger.log_event("WARN_USER", user_id, {"reason": reason})

    @staticmethod
    def remove_content(user_id: str, content_id: str, reason: str) -> None:
        AuditLogger.log_event("REMOVE_CONTENT", user_id, {"content_id": content_id, "reason": reason})

    @staticmethod
    def suspend_account(user_id: str, duration_days: int, reason: str) -> None:
        AuditLogger.log_event(
            "SUSPEND_ACCOUNT", user_id, {"duration_days": duration_days, "reason": reason}
        )

    @staticmethod
    def escalate(user_id: str, reason: str, admin_id: Optional[str] = None) -> None:
        AuditLogger.log_event("ESCALATE", user_id, {"reason": reason, "admin": admin_id})


class EnforcementEngine:
    """Engine for handling moderation violations."""

    @staticmethod
    def handle_violation(user_id: str, content_id: str, violation: ModerationViolation) -> None:
        reason = violation.message

        EnforcementAction.remove_content(user_id, content_id, reason)

        if "HATE" in reason or "VIOLENCE" in reason:
            EnforcementAction.suspend_account(user_id, 30, reason)
            EnforcementAction.escalate(user_id, reason)
        else:
            EnforcementAction.warn_user(user_id, reason)
