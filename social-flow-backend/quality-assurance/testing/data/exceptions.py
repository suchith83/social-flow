"""
Custom exceptions for the data package.
"""

class DataError(Exception):
    """Base class for data package errors."""


class SchemaValidationError(DataError):
    """Raised when data fails schema validation."""


class FixtureLoadError(DataError):
    """Raised when fixture file cannot be loaded or parsed."""


class GeneratorError(DataError):
    """Raised for errors during generation of test data."""
