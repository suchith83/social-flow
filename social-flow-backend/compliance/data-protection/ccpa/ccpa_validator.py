"""
# Validate CCPA requests and compliance rules
"""
"""
CCPA Request Validator
----------------------
Validates incoming consumer requests to ensure they meet
CCPA requirements.
"""

from .ccpa_exceptions import CCPAValidationError

class CCPAValidator:
    """Validator for CCPA compliance requests."""

    REQUIRED_FIELDS = {
        "access": {"user_id", "request_id"},
        "deletion": {"user_id", "request_id"},
        "optout": {"user_id", "request_id"},
    }

    @classmethod
    def validate_request(cls, request: dict, request_type: str):
        """
        Ensure request has all required fields.
        """
        missing = cls.REQUIRED_FIELDS.get(request_type, set()) - request.keys()
        if missing:
            raise CCPAValidationError(f"Missing fields: {', '.join(missing)}")
        return True
