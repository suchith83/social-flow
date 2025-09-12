"""
# Custom exceptions for GDPR flows
"""
"""
Custom Exceptions for GDPR Compliance
-------------------------------------
"""

class GDPRRequestError(Exception):
    """Raised when a DSAR (Data Subject Access Request) fails."""
    pass


class GDPRValidationError(Exception):
    """Raised when a GDPR request fails validation checks."""
    pass


class GDPRConsentError(Exception):
    """Raised when consent handling fails."""
    pass


class GDPRAuditError(Exception):
    """Raised when an audit logging operation fails."""
    pass


class GDPRPortabilityError(Exception):
    """Raised when exporting user data fails."""
    pass
