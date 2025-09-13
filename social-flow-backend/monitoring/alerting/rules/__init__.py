# Package initializer for rules module
"""
Alerting Rules package.

Exports:
- Rule and Condition primitives
- RuleEngine to evaluate events and execute actions
- Builtin actions: notify, create_incident, silence
- Template utilities and validators
"""

from .rule import Rule, RuleMatchContext
from .conditions import Condition, AndCondition, OrCondition, ComparisonCondition, ExistsCondition
from .actions import notify_action, create_incident_action, silence_action, ActionResult, ActionContext
from .engine import RuleEngine
from .templates import render_template
from .validator import validate_rule
from .persistence import InMemoryRuleStore, RuleStore

__all__ = [
    "Rule",
    "RuleMatchContext",
    "Condition",
    "AndCondition",
    "OrCondition",
    "ComparisonCondition",
    "ExistsCondition",
    "notify_action",
    "create_incident_action",
    "silence_action",
    "ActionResult",
    "ActionContext",
    "RuleEngine",
    "render_template",
    "validate_rule",
    "InMemoryRuleStore",
    "RuleStore",
]
