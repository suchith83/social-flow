"""
Custom exceptions used by strategies package.
"""


class StrategyError(Exception):
    """Base class for strategy-related errors."""


class StrategyValidationError(StrategyError):
    """Raised when provided strategy config/plan is invalid."""


class EnforcementError(StrategyError):
    """Raised when policy enforcement fails unexpectedly."""
