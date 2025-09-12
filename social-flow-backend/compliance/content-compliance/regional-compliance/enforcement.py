"""
enforcement.py

Orchestrates enforcement actions based on regional policy decisions.
Integrates with previously defined enforcement modules (moderation/age/takedown).
This file focuses on region-aware enforcement decisions, audit logging, and escalation thresholds.
"""

from typing import Dict, Any, Optional
from .audit_logger import RegionalAuditLogger
from .regional_policy import RegionalPolicyRegistry
from .legal_requirements import get_legal_requirements_for_jurisdiction
import logging

logger = logging.getLogger(__name__)

class RegionalEnforcement:
    """
    Facade to perform enforcement actions in a jurisdiction-aware manner.
    It logs actions via RegionalAuditLogger and decides whether to auto-enforce or escalate.
    """

    def __init__(self):
        # threshold mapping may be pulled from dynamic config in real system
        self.escalation_severity_threshold = 75

    def enforce(self, jurisdiction: str, action: str, actor: str, target: Dict[str, Any], reason: str, severity: int = 0) -> None:
        """
        action: e.g., "block_content", "quarantine_data", "migrate_data", "takedown"
        target: metadata about target (user_id, content_id, storage_region)
        """
        # Check local rules if enforcement permitted
        rules = RegionalPolicyRegistry.get_rules_for_jurisdiction(jurisdiction)
        # For demo: we log and decide escalate if severity high or rule explicitly requires escalation
        RegionalAuditLogger.log_event("ENFORCEMENT_ATTEMPT", actor, {"jurisdiction": jurisdiction, "action": action, "target": target, "reason": reason, "severity": severity})
        # Basic escalation decision
        if severity >= self.escalation_severity_threshold:
            RegionalAuditLogger.log_event("ENFORCEMENT_ESCALATED", "system", {"jurisdiction": jurisdiction, "action": action, "target": target, "reason": reason})
            # Real system: call legal/ticketing here
        # Real system: call enforcement APIs (content service/data service)
        RegionalAuditLogger.log_event("ENFORCEMENT_COMPLETED", actor, {"jurisdiction": jurisdiction, "action": action, "target": target})

    def check_and_enforce_content(self, user_profile: Dict[str, Any], content_meta: Dict[str, Any], compliance_result: Dict[str, Any], actor: str) -> None:
        """
        Convenience method: apply decisions from ComplianceChecker.evaluate_content
        """
        jurisdiction = user_profile.get("jurisdiction", "global")
        allowed = compliance_result.get("allowed", True)
        if not allowed:
            # choose first suggested action
            actions = compliance_result.get("actions", [])
            severity = compliance_result.get("severity", 0)
            action = actions[0] if actions else "block"
            self.enforce(jurisdiction, action, actor, {"content": content_meta, "user": user_profile}, "; ".join(compliance_result.get("reasons", [])), severity)
        else:
            RegionalAuditLogger.log_event("NO_ACTION_REQUIRED", actor, {"jurisdiction": jurisdiction, "content": content_meta})
