# Escalation policy definitions and rules
"""
Escalation policy definitions.

EscalationPolicy contains ordered escalation levels. Each level describes:
- who/what to notify (e.g., contact ids or channel descriptors)
- timeout (how long to wait before moving to next level)
- retries/backoff configuration

This module does not perform sending — it only models policies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import timedelta


@dataclass
class EscalationLevel:
    """
    One level within an escalation policy.

    Attributes:
        name: human-readable name, e.g., "oncall", "team-leads", "pagerduty"
        targets: list of target descriptors (strings or dicts) that Escalator understands.
        timeout: timedelta to wait before escalating to the next level.
        retries: number of retries for this level before escalting.
        retry_backoff: optional backoff policy name or dict used by escalator/retry logic.
    """
    name: str
    targets: List[Dict[str, Any]]  # E.g. [{"type":"slack","webhook": "..."}, {"type":"sms","to":"+..."}]
    timeout: timedelta = timedelta(minutes=5)
    retries: int = 0
    retry_backoff: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """
    An ordered policy of escalation levels.

    Example:
        policy = EscalationPolicy(
            name="critical-db",
            levels=[
                EscalationLevel(name="primary-oncall", targets=[...], timeout=timedelta(minutes=2)),
                EscalationLevel(name="secondary-oncall", targets=[...], timeout=timedelta(minutes=5)),
                EscalationLevel(name="pagerduty", targets=[...], timeout=timedelta(minutes=0)),
            ],
        )
    """
    name: str
    levels: List[EscalationLevel]

    def get_level(self, index: int) -> EscalationLevel:
        """Returns the EscalationLevel at index; raises IndexError if out-of-range."""
        return self.levels[index]

    def num_levels(self) -> int:
        return len(self.levels)
