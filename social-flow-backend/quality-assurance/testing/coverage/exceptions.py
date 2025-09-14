"""
Custom exception classes for coverage package.
"""


class CoverageError(Exception):
    """Base exception for coverage issues."""


class CoverageThresholdError(CoverageError):
    """Raised when coverage falls below configured threshold."""


class CoverageIntegrationError(CoverageError):
    """Raised for CI/CD integration related issues."""
