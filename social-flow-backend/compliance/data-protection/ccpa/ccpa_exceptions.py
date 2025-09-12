"""
Custom Exceptions for CCPA Compliance
-------------------------------------
These exceptions provide a structured way to handle errors
related to CCPA compliance.
"""

class CCPARequestError(Exception):
    """Raised when a Data Subject Access Request (DSAR) fails."""
    pass


class CCPAValidationError(Exception):
    """Raised when a request fails validation checks."""
    pass


class CCPANotificationError(Exception):
    """Raised when user notification fails."""
    pass


class CCPAAuditError(Exception):
    """Raised when an audit logging operation fails."""
    pass
