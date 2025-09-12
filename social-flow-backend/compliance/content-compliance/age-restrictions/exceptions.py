"""
exceptions.py

Custom exceptions for age restriction compliance.
"""


class AgeRestrictionViolation(Exception):
    """Raised when a user tries to access restricted content."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PolicyConfigurationError(Exception):
    """Raised when policy definitions are invalid or inconsistent."""
    pass
