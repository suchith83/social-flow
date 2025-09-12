"""
integration.py

Integration layer exposing APIs for services to use age restriction compliance.
"""

from datetime import date
from typing import Dict

from .validator import AgeValidator
from .age_policy import ContentCategory, Jurisdiction
from .exceptions import AgeRestrictionViolation
from .enforcement import EnforcementEngine
from .config import ComplianceConfig


class AgeRestrictionService:
    """Public API service for age restriction checks."""

    def __init__(self):
        self.violation_counts: Dict[str, int] = {}

    def check_content_access(
        self,
        user_id: str,
        birthdate: date,
        category: ContentCategory,
        jurisdiction: Jurisdiction = Jurisdiction.GLOBAL,
    ) -> bool:
        """Main API to check if user can access content."""
        try:
            AgeValidator.validate_access(birthdate, category, jurisdiction)
            return True
        except AgeRestrictionViolation as violation:
            self._record_violation(user_id)
            EnforcementEngine.handle_violation(user_id, violation)
            return False

    def _record_violation(self, user_id: str) -> None:
        """Track violations per user for escalation."""
        self.violation_counts[user_id] = self.violation_counts.get(user_id, 0) + 1
        if self.violation_counts[user_id] >= ComplianceConfig.ESCALATION_THRESHOLD:
            EnforcementEngine.escalate(
                user_id, f"Exceeded violation threshold: {self.violation_counts[user_id]}"
            )
