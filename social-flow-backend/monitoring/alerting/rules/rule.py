# Defines Rule data structures
"""
Rule model.

A Rule contains:
- id, name, description
- a Condition (tree) that is evaluated against an incoming event/context
- one or more actions to execute when the rule matches
- enabled/disabled flag and optional mute/silence window
- priority and metadata
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Callable, Optional
import uuid

from .conditions import Condition

@dataclass
class RuleMatchContext:
    """
    Context passed to actions when a rule matches.
    - event: raw event/metric dict
    - matched_values: optional values extracted by conditions
    - rule: the Rule that matched
    - timestamp: when matched
    """
    event: Dict[str, Any]
    matched_values: Dict[str, Any]
    rule: "Rule"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Rule:
    """
    Rule dataclass.
    Fields:
      - id: unique id (uuid)
      - name: human friendly name
      - description: optional
      - condition: Condition instance
      - actions: list of callables (Action factories or descriptors) or action dicts
      - enabled: whether rule is active
      - created_at, updated_at
      - mute_until: optional datetime; if now < mute_until rule is silenced
      - priority: numeric priority (higher -> more urgent)
      - metadata: free-form dict
    """
    name: str
    condition: Condition
    actions: List[Dict[str, Any]]  # action descriptors (see actions.py) or callables
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mute_until: Optional[datetime] = None
    priority: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_silenced(self, at_time: Optional[datetime] = None) -> bool:
        from datetime import datetime, timezone
        at_time = at_time or datetime.now(timezone.utc)
        return self.mute_until is not None and at_time < self.mute_until
