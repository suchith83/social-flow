"""
validator.py

Validation engine for checking if users meet the minimum age requirement
based on content category and jurisdiction.
"""

from datetime import date
from typing import Optional

from .age_policy import AgePolicy, ContentCategory, Jurisdiction
from .exceptions import AgeRestrictionViolation


class AgeValidator:
    """Validator for user age against policies."""

    @staticmethod
    def calculate_age(birthdate: date, today: Optional[date] = None) -> int:
        """Utility to calculate age from date of birth."""
        today = today or date.today()
        age = today.year - birthdate.year - (
            (today.month, today.day) < (birthdate.month, birthdate.day)
        )
        return age

    @classmethod
    def validate_access(
        cls,
        birthdate: date,
        category: ContentCategory,
        jurisdiction: Jurisdiction = Jurisdiction.GLOBAL,
    ) -> None:
        """
        Validates if user with birthdate is allowed to access category content.
        Raises AgeRestrictionViolation if not permitted.
        """
        user_age = cls.calculate_age(birthdate)
        required_age = AgePolicy.get_required_age(category, jurisdiction)

        if user_age < required_age:
            raise AgeRestrictionViolation(
                f"Access denied: User age {user_age} is below required {required_age} "
                f"for {category.value} content in {jurisdiction.value.upper()}."
            )
