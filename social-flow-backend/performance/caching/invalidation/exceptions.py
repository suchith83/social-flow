# exceptions.py
# Created by Create-Invalidation.ps1
"""Custom exceptions used by the invalidation package."""

class InvalidationError(Exception):
    """Base exception for invalidation errors."""
    pass


class KeyBuildError(InvalidationError):
    """Raised when a cache key cannot be built."""
    pass


class RemoteInvalidateError(InvalidationError):
    """Raised when a remote invalidation fails (e.g., Redis, CDN)."""
    pass
