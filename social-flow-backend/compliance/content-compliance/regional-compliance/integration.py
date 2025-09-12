"""
integration.py

Public API used by other platform services:
- determine jurisdiction
- run compliance checks for content and data residency
- run region-aware enforcement (delegates to enforcement.py)

This file is the recommended integration surface for other backend services.
"""

from typing import Dict, Any, Optional
from .jurisdiction import resolve_jurisdiction_from_user
from .compliance_checker import ComplianceChecker
from .enforcement import RegionalEnforcement
from .audit_logger import RegionalAuditLogger
import logging

logger = logging.getLogger(__name__)

class RegionalComplianceService:
    def __init__(self):
        self.checker = ComplianceChecker()
        self.enforcer = RegionalEnforcement()

    def determine_jurisdiction(self, user_profile: Dict[str, Any], ip_geolocation: Optional[Dict[str, Any]] = None) -> str:
        j = resolve_jurisdiction_from_user({**user_profile, "ip_geolocation": ip_geolocation})
        RegionalAuditLogger.log_event("JURISDICTION_RESOLVED", "system", {"resolved": j, "user_profile": user_profile})
        return j

    def check_content(self, user_profile: Dict[str, Any], content_meta: Dict[str, Any], actor: str = "system") -> Dict[str, Any]:
        """
        Evaluate and (optionally) enforce content decisions.
        Returns structured decision object.
        """
        # canonicalize jurisdiction to profile for downstream logic
        jurisdiction = self.determine_jurisdiction(user_profile, user_profile.get("ip_geolocation"))
        user_profile = {**user_profile, "jurisdiction": jurisdiction}
        result = self.checker.evaluate_content(user_profile, content_meta)
        RegionalAuditLogger.log_event("COMPLIANCE_CHECK", actor, {"jurisdiction": jurisdiction, "result": result.to_dict(), "content_meta": content_meta})
        # If blocked, call enforcement
        if not result.allowed:
            self.enforcer.check_and_enforce_content(user_profile, content_meta, result.to_dict(), actor)
        return result.to_dict()

    def check_data_residency(self, user_profile: Dict[str, Any], data_location: Optional[str], actor: str = "system") -> Dict[str, Any]:
        jurisdiction = self.determine_jurisdiction(user_profile, user_profile.get("ip_geolocation"))
        user_profile = {**user_profile, "jurisdiction": jurisdiction}
        result = self.checker.evaluate_data_residency(user_profile, data_location)
        RegionalAuditLogger.log_event("DATA_RESIDENCY_CHECK", actor, {"jurisdiction": jurisdiction, "result": result.to_dict(), "data_location": data_location})
        if not result.allowed:
            # enforcement could trigger migrations/holds/quarantine
            self.enforcer.enforce(jurisdiction, "quarantine_data", actor, {"data_location": data_location, "user": user_profile}, "; ".join(result.reasons), result.severity)
        return result.to_dict()
