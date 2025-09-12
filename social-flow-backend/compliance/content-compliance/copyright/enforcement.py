"""
enforcement.py

High-level enforcement actions and orchestration with the takedown manager and audit logs.
Implements soft removals, hard deletes (admin), account penalties, and legal escalation hooks.
"""

from typing import Optional
from .takedown_manager.py import TakedownManager  # note: safe relative import in package
from .audit_logger import AuditLogger
from .exceptions import EscalationError
from .legal_escalation import LegalEscalation

# For demonstration: actions are logged; in production they call services.


class EnforcementService:
    """Facade for enforcement actions."""

    @staticmethod
    def soft_remove(content_id: str, actor: str, reason: str) -> None:
        """Mark content hidden / soft-removed (non-destructive)."""
        # stub: In real system call content service to set visibility flag
        AuditLogger.log_event("SOFT_REMOVE", actor, {"content_id": content_id, "reason": reason})

    @staticmethod
    def hard_delete(content_id: str, actor: str, reason: str) -> None:
        """Permanently remove content (admin-only)."""
        AuditLogger.log_event("HARD_DELETE", actor, {"content_id": content_id, "reason": reason})
        # in production: queue deletion job, remove from CDN, purge caches, revoke downloads, etc.

    @staticmethod
    def penalize_account(user_id: str, actor: str, duration_days: int, reason: str) -> None:
        """Apply account-level penalties like temporary suspension."""
        AuditLogger.log_event("PENALIZE_ACCOUNT", actor, {"user_id": user_id, "duration_days": duration_days, "reason": reason})

    @staticmethod
    def escalate_to_legal(notice_id: str, details: dict) -> None:
        """Escalate case to legal team via LegalEscalation helper."""
        try:
            LegalEscalation.notify_legal(notice_id, details)
            AuditLogger.log_event("ESCALATED_TO_LEGAL", "system", {"notice_id": notice_id})
        except Exception as e:
            raise EscalationError(str(e))
