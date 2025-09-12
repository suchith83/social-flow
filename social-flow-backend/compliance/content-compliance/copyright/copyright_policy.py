"""
copyright_policy.py

Defines platform copyright policy, categories, severity and mapping to enforcement actions.
This centralizes rules and makes it easy to adjust policy by region or business needs.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class CopyrightCategory(Enum):
    """High-level categories used to classify reported items."""
    DIRECT_INFRINGEMENT = "direct_infringement"    # Copied original content
    MODIFIED_INFRINGEMENT = "modified_infringement"  # Minor edits of original
    PARODY_FLAW = "parody"                          # Potential fair use
    LICENSE_VIOLATION = "license_violation"        # Using content beyond license
    POTENTIAL_FAIR_USE = "potential_fair_use"      # May be protected
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EnforcementRule:
    """Defines an enforcement rule for a category."""
    remove_immediately: bool
    escalate_to_legal: bool
    severity_score: int  # numeric severity used for aggregation/thresholds


# Default rules mapping. Can be overridden by jurisdiction or dynamic config.
_DEFAULT_RULES: Dict[CopyrightCategory, EnforcementRule] = {
    CopyrightCategory.DIRECT_INFRINGEMENT: EnforcementRule(True, True, 90),
    CopyrightCategory.MODIFIED_INFRINGEMENT: EnforcementRule(True, True, 80),
    CopyrightCategory.LICENSE_VIOLATION: EnforcementRule(True, False, 70),
    CopyrightCategory.POTENTIAL_FAIR_USE: EnforcementRule(False, True, 40),
    CopyrightCategory.PARODY_FLAW: EnforcementRule(False, True, 30),
    CopyrightCategory.UNKNOWN: EnforcementRule(False, False, 10),
}


class CopyrightPolicy:
    """Policy accessor and exporter for copyright flows."""

    _rules: Dict[CopyrightCategory, EnforcementRule] = _DEFAULT_RULES.copy()

    @classmethod
    def get_rule(cls, category: CopyrightCategory) -> EnforcementRule:
        return cls._rules.get(category, _DEFAULT_RULES[CopyrightCategory.UNKNOWN])

    @classmethod
    def export_policy(cls) -> Dict[str, Any]:
        return {
            category.value: {
                "remove_immediately": rule.remove_immediately,
                "escalate_to_legal": rule.escalate_to_legal,
                "severity_score": rule.severity_score,
            }
            for category, rule in cls._rules.items()
        }

    @classmethod
    def override_rules(cls, overrides: Dict[CopyrightCategory, EnforcementRule]) -> None:
        """
        Replace or add to policy rules at runtime.
        This call is expected to be used only by secure admin tooling.
        """
        cls._rules.update(overrides)
