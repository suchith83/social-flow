"""
exceptions.py

Custom exceptions for moderation system.
"""


class ModerationViolation(Exception):
    """Raised when user content violates moderation rules."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PolicyError(Exception):
    """Raised for invalid or missing moderation policies."""
    pass
