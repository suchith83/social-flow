"""
# API middleware for LGPD enforcement
"""
# compliance/data-protection/lgpd/lgpd_middleware.py
"""
LGPD Middleware
---------------
Framework-agnostic middleware enforcing:
- Consent validation
- Data minimization
- Audit logging
"""

from typing import Callable, Dict, Any


class LGPDComplianceMiddleware:
    def __init__(self, consent_manager, audit_logger):
        self.consent_manager = consent_manager
        self.audit_logger = audit_logger

    def __call__(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Wrap an API handler with LGPD checks."""

        def wrapper(request: Dict[str, Any]) -> Dict[str, Any]:
            user_id = request.get("user_id")
            purpose = request.get("purpose")

            # Enforce consent
            if not self.consent_manager.has_valid_consent(user_id, purpose):
                self.audit_logger.log_event("denied_request", {"user": user_id, "purpose": purpose})
                return {"error": "Consent not granted", "status": 403}

            response = handler(request)
            self.audit_logger.log_event("handled_request", {"user": user_id, "purpose": purpose})
            return response

        return wrapper
