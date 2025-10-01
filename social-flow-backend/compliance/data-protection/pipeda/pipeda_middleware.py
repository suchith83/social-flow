"""
# API middleware for PIPEDA enforcement
"""
# compliance/data-protection/pipeda/pipeda_middleware.py
"""
PIPEDA Middleware
-----------------
Framework-agnostic middleware enforcing:
- Consent validation
- Audit logging
- Purpose limitation
"""

from typing import Callable, Dict, Any


class PIPEDAMiddleware:
    def __init__(self, consent_manager, audit_logger):
        self.consent_manager = consent_manager
        self.audit_logger = audit_logger

    def __call__(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        def wrapper(request: Dict[str, Any]) -> Dict[str, Any]:
            user_id = request.get("user_id")
            purpose = request.get("purpose")

            if not self.consent_manager.has_consent(user_id, purpose):
                self.audit_logger.log_event("denied_request", {"user": user_id, "purpose": purpose})
                return {"error": "Consent not obtained", "status": 403}

            response = handler(request)
            self.audit_logger.log_event("processed_request", {"user": user_id, "purpose": purpose})
            return response
        return wrapper
