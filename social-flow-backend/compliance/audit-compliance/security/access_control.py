# access_control.py
from typing import Dict, List, Optional, Any
from .utils import log_event, timing_safe_compare, secure_hash
import fnmatch
import re

# Simple policy-as-code data structure examples
# Policies are a list of dicts:
# {
#   "id": "policy-1",
#   "description": "Admins may manage everything",
#   "effect": "allow" | "deny",
#   "subjects": ["role:admin", "user:alice"],
#   "actions": ["*"] or ["read","write","delete"],
#   "resources": ["*"] or ["project/*", "s3:bucket/*"],
#   "conditions": {"time": {"gte": "09:00", "lte": "18:00"}, "ip": "10.0.0.*"}
# }

class AccessControlEvaluator:
    """
    Evaluates RBAC/ABAC-like policies using a policy-as-code approach.
    - Supports wildcard patterns and simple conditions (time/ip/attrs).
    - Evaluates policies in deterministic order (deny overrides allow).
    """

    def __init__(self, policies: List[Dict[str, Any]]):
        self.policies = policies

    @staticmethod
    def _match_pattern(pattern: str, value: str) -> bool:
        # Support glob-like pattern matching and regex patterns prefixed with 're:'
        if pattern.startswith("re:"):
            return re.search(pattern[3:], value) is not None
        return fnmatch.fnmatch(value, pattern)

    def _subject_matches(self, policy_subjects: List[str], subject: str) -> bool:
        return any(self._match_pattern(p, subject) for p in policy_subjects)

    def _action_matches(self, policy_actions: List[str], action: str) -> bool:
        return any(self._match_pattern(p, action) for p in policy_actions)

    def _resource_matches(self, policy_resources: List[str], resource: str) -> bool:
        return any(self._match_pattern(p, resource) for p in policy_resources)

    def _conditions_match(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Implement a few condition checks like time window and ip glob
        if not conditions:
            return True
        # time condition: expects 'gte' and/or 'lte' in HH:MM format compared to context['time']
        time_cond = conditions.get("time")
        if time_cond and "time" in context:
            now = context["time"]  # assume "HH:MM" formatted string
            if "gte" in time_cond and now < time_cond["gte"]:
                return False
            if "lte" in time_cond and now > time_cond["lte"]:
                return False
        ip_cond = conditions.get("ip")
        if ip_cond and "ip" in context:
            if not fnmatch.fnmatch(context["ip"], ip_cond):
                return False
        # other attribute checks: simple equality
        for k, v in conditions.items():
            if k in ("time", "ip"):
                continue
            if k in context and context[k] != v:
                return False
        return True

    def is_allowed(self, subject: str, action: str, resource: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if `subject` can perform `action` on `resource` in `context`.
        - Policy evaluation order: explicit deny takes precedence.
        - If no policy matches, default deny.
        """
        context = context or {}
        allow_found = False
        for p in self.policies:
            if not self._subject_matches(p.get("subjects", ["*"]), subject):
                continue
            if not self._action_matches(p.get("actions", ["*"]), action):
                continue
            if not self._resource_matches(p.get("resources", ["*"]), resource):
                continue
            if not self._conditions_match(p.get("conditions", {}), context):
                continue
            effect = p.get("effect", "allow").lower()
            log_event(f"Policy matched: {p.get('id')} effect={effect}", level="INFO", policy_id=p.get("id"), subject=subject)
            if effect == "deny":
                log_event("Access denied by policy", level="WARNING", subject=subject, resource=resource, action=action)
                return False
            if effect == "allow":
                allow_found = True
        if allow_found:
            log_event("Access allowed (no deny match)", level="INFO", subject=subject, resource=resource, action=action)
            return True
        log_event("Access denied (no matching allow policy)", level="WARNING", subject=subject, resource=resource, action=action)
        return False

    # Utility helpers for policy authoring and fingerprinting
    @staticmethod
    def fingerprint_policy(policy: Dict[str, Any]) -> str:
        """Return a stable fingerprint for a policy (hash of canonical JSON)."""
        canonical = json_deterministic(policy)
        return secure_hash(canonical, salt="policy-salt")

def json_deterministic(obj: Any) -> str:
    """Return deterministic JSON string for hashing (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
