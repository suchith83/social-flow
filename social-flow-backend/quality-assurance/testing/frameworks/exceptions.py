"""
Custom exceptions for the frameworks package.
"""


class FrameworksError(Exception):
    """Base error for frameworks package."""


class AdapterNotFoundError(FrameworksError):
    """Raised when an adapter for a framework is not available."""


class TestExecutionError(FrameworksError):
    """Raised when the test run fails in an unexpected way."""
