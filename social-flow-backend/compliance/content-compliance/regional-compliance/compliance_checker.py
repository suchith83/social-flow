"""
compliance_checker.py

Core engine to evaluate actions/content against regional rules.
Supports:
- age checks vs regional policy (ties into age-restrictions)
- data residency checks
- retention & legal hold checks
- region-specific moderation overrides
Returns structured results with reasons, severity, and suggested enforcement actions.
"""

from typing import Dict, Any, Optional, List
from .regional_policy import RegionalPolicyRegistry, PolicyScope
from .jurisdiction import resolve_jurisdiction_from_user
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ComplianceResult:
    def __init__(self, allowed: bool, reasons: Optional[List[str]] = None, actions: Optional[List[str]] = None, severity: int = 0):
        self.allowed = allowed
        self.reasons = reasons or []
        self.actions = actions or []
        self.severity = severity

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "reasons": self.reasons,
            "actions": self.actions,
            "severity": self.severity
        }

class ComplianceChecker:
    """Evaluate various compliance dimensions for a given context."""

    def __init__(self):
        pass

    def evaluate_content(self, user_profile: Dict[str, Any], content_meta: Dict[str, Any]) -> ComplianceResult:
        """
        Evaluate whether content is allowed for a user in their resolved jurisdiction.
        content_meta includes: {category, content_type, tags, declared_age_restriction}
        """
        jurisdiction = resolve_jurisdiction_from_user(user_profile, user_profile.get("ip_geolocation"))
        rules = RegionalPolicyRegistry.get_rules_for_jurisdiction(jurisdiction)
        reasons = []
        actions = []
        severity = 0
        allowed = True

        # Example: rule-based min_age checks (if rule exists)
        # Look for a data-driven rule like "<something>_min_age" in CONTENT scope
        for rid, r in rules.items():
            if r["scope"] != PolicyScope.CONTENT.value and r["scope"] != PolicyScope.GLOBAL.value:
                continue
            params = r.get("params", {})
            # supports a generic "applies_to" param to match content categories
            applies = params.get("applies_to")
            if applies and content_meta.get("category") not in applies:
                continue
            min_age = params.get("min_age")
            if min_age:
                # user_profile must contain 'birthdate' in ISO format or 'declared_age'
                user_age = None
                if user_profile.get("declared_age") is not None:
                    user_age = int(user_profile["declared_age"])
                elif user_profile.get("birthdate"):
                    try:
                        bd = datetime.fromisoformat(user_profile["birthdate"]).date()
                        today = datetime.utcnow().date()
                        user_age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
                    except Exception:
                        user_age = None
                if user_age is None or user_age < min_age:
                    allowed = False
                    reasons.append(f"Policy {rid} requires min_age {min_age} for {content_meta.get('category')}")
                    actions.append("block")
                    severity = max(severity, params.get("severity", 50))
        # Data residency checks (example)
        for rid, r in rules.items():
            if r["scope"] != PolicyScope.DATA.value:
                continue
            params = r.get("params", {})
            required_region = params.get("required_region")
            if required_region:
                storage_hint = content_meta.get("storage_region")
                if storage_hint and storage_hint.lower() != required_region.lower():
                    allowed = False
                    reasons.append(f"Policy {rid} requires storage in {required_region}; found {storage_hint}")
                    actions.append("quarantine")
                    severity = max(severity, params.get("severity", 60))

        # Default: if no reasons to block, allowed True
        return ComplianceResult(allowed=allowed, reasons=reasons, actions=actions, severity=severity)

    def evaluate_data_residency(self, user_profile: Dict[str, Any], data_location: Optional[str]) -> ComplianceResult:
        """
        Check whether storing user's data in `data_location` conflicts with regional rules.
        """
        jurisdiction = resolve_jurisdiction_from_user(user_profile, user_profile.get("ip_geolocation"))
        rules = RegionalPolicyRegistry.get_rules_for_jurisdiction(jurisdiction)
        reasons = []
        actions = []
        allowed = True
        severity = 0

        for rid, r in rules.items():
            if r["scope"] != PolicyScope.DATA.value:
                continue
            params = r.get("params", {})
            required_region = params.get("required_region")
            if required_region:
                if not data_location:
                    # Unknown storage location => not allowed by default
                    allowed = False
                    reasons.append(f"Policy {rid} requires region {required_region} but storage location unknown")
                    actions.append("hold")
                    severity = max(severity, params.get("severity", 70))
                elif data_location.lower() != required_region.lower():
                    allowed = False
                    reasons.append(f"Policy {rid} requires region {required_region} but storing in {data_location}")
                    actions.append("migrate")
                    severity = max(severity, params.get("severity", 70))
        return ComplianceResult(allowed=allowed, reasons=reasons, actions=actions, severity=severity)
