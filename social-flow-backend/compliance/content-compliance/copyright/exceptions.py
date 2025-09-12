"""
exceptions.py

Custom exceptions used across the copyright module.
"""

class CopyrightError(Exception):
    """Base exception for copyright module."""
    pass


class InvalidNoticeError(CopyrightError):
    """Raised when a submitted takedown/notice is malformed or invalid."""
    pass


class EvidenceNotFoundError(CopyrightError):
    """Raised when requested evidence cannot be located."""
    pass


class TakedownAlreadyProcessedError(CopyrightError):
    """Raised when an attempt is made to process a takedown that is closed."""
    pass


class EscalationError(CopyrightError):
    """Raised when escalation to legal/admin fails."""
    pass
