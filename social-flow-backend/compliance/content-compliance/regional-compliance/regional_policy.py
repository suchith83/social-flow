"""
regional_policy.py

Registry and management for regional (jurisdictional) compliance rules.
Designed to be dynamically updatable (admin-only) and queried by the compliance checker.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from enum import Enum
import threading
import copy
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

# Thread-safe in-memory registry for demo. Replace with DB/cache in production.
_REGISTRY_LOCK = threading.RLock()
_REGISTRY: Dict[str, Dict[str, Any]] = {}

class PolicyScope(Enum):
    GLOBAL = "global"
    CONTENT = "content"
    DATA = "data"
    USER = "user"

@dataclass
class RegionalRule:
    """A single rule applying to a jurisdiction."""
    id: str
    jurisdiction: str                               # e.g., "us", "eu", "india"
    scope: PolicyScope
    description: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)   # rule parameters (e.g., min_age)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RegionalPolicyRegistry:
    """Manage the set of rules for all jurisdictions."""
    @staticmethod
    def register_rule(rule: RegionalRule) -> None:
        with _REGISTRY_LOCK:
            j = rule.jurisdiction.lower()
            _REGISTRY.setdefault(j, {})[rule.id] = rule.to_dict()
            logger.info(f"Registered regional rule {rule.id} for {j}")

    @staticmethod
    def update_rule(jurisdiction: str, rule_id: str, patch: Dict[str, Any]) -> None:
        with _REGISTRY_LOCK:
            j = jurisdiction.lower()
            if j not in _REGISTRY or rule_id not in _REGISTRY[j]:
                raise KeyError(f"Rule {rule_id} not found for {j}")
            _REGISTRY[j][rule_id].update(patch)
            logger.info(f"Updated regional rule {rule_id} for {j}")

    @staticmethod
    def remove_rule(jurisdiction: str, rule_id: str) -> None:
        with _REGISTRY_LOCK:
            j = jurisdiction.lower()
            if j in _REGISTRY and rule_id in _REGISTRY[j]:
                del _REGISTRY[j][rule_id]
                logger.info(f"Removed regional rule {rule_id} for {j}")

    @staticmethod
    def get_rules_for_jurisdiction(jurisdiction: str) -> Dict[str, Dict[str, Any]]:
        with _REGISTRY_LOCK:
            return copy.deepcopy(_REGISTRY.get(jurisdiction.lower(), {}))

    @staticmethod
    def export_registry() -> Dict[str, Dict[str, Any]]:
        with _REGISTRY_LOCK:
            return copy.deepcopy(_REGISTRY)


# Seed some realistic sample rules (can be overridden)
def _seed_defaults():
    RegionalPolicyRegistry.register_rule(RegionalRule(
        id="eu_data_residency",
        jurisdiction="eu",
        scope=PolicyScope.DATA,
        description="EU personal data should be stored in EU-located storage unless consented.",
        params={"required_region": "eu", "exception_with_consent": True},
        created_by="system"
    ))
    RegionalPolicyRegistry.register_rule(RegionalRule(
        id="us_age_limit_video_gambling",
        jurisdiction="us",
        scope=PolicyScope.CONTENT,
        description="Gambling content requires 21+ in certain states (example).",
        params={"default_min_age": 18, "state_overrides": {"nevada": 21}},
        created_by="system"
    ))
    RegionalPolicyRegistry.register_rule(RegionalRule(
        id="india_retention_personal",
        jurisdiction="india",
        scope=PolicyScope.DATA,
        description="Certain personal data retention minima for legal investigations.",
        params={"min_retention_days": 180},
        created_by="system"
    ))

_seed_defaults()
