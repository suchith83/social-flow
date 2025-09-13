# Validates rules for syntax and constraints
"""
Lightweight schema/consistency validator for Rule objects.

Checks:
- condition is a Condition instance
- actions are known types and have required params
- name present and unique check delegated to store
"""

from typing import Dict, Any
import logging

from .rule import Rule
from .conditions import Condition
from .actions import notify_action, create_incident_action, silence_action

logger = logging.getLogger(__name__)

KNOWN_ACTIONS = {"notify", "create_incident", "silence"}
ACTION_REQUIRED_PARAMS = {
    "notify": ["targets"],
    "create_incident": [],
    "silence": ["duration_seconds"]
}

def validate_rule(rule: Rule, store=None) -> None:
    if not rule.name or not isinstance(rule.name, str):
        raise ValueError("Rule must have a name")
    if not isinstance(rule.condition, Condition):
        raise ValueError("Rule condition must be a Condition instance")
    if not isinstance(rule.actions, list) or len(rule.actions) == 0:
        raise ValueError("Rule must have at least one action descriptor")
    # validate actions
    for a in rule.actions:
        if not isinstance(a, dict) or "type" not in a:
            raise ValueError("Each action must be a dict with 'type' key")
        t = a["type"]
        if t not in KNOWN_ACTIONS:
            raise ValueError(f"Unknown action type: {t}")
        required = ACTION_REQUIRED_PARAMS.get(t, [])
        params = a.get("params", {})
        for r in required:
            if r not in params:
                raise ValueError(f"Action {t} missing required param: {r}")
    # optional uniqueness check
    if store:
        existing = store.get_rule_by_name(rule.name)
        if existing and existing.id != rule.id:
            raise ValueError(f"Rule name must be unique: {rule.name}")
