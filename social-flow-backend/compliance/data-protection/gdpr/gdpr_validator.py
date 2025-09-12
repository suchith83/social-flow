"""
# Validators for GDPR compliance checks
"""
"""
GDPR Request Validator
----------------------
"""

from .gdpr_exceptions import GDPRValidationError

class GDPRValidator:
    """Validator for GDPR request payloads."""

    REQUIRED_FIELDS = {
        "access": {"user_id", "request_id"},
        "deletion": {"user_id", "request_id"},
        "portability": {"user_id", "request_id"},
        "consent": {"user_id", "request_id"},
    }

    @classmethod
    def validate_request(cls, request: dict, request_type: str):
        missing = cls.REQUIRED_FIELDS.get(request_type, set()) - request.keys()
        if missing:
            raise GDPRValidationError(f"Missing fields: {', '.join(missing)}")
        return True
