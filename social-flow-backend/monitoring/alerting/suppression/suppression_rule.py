# Defines suppression rule data structures
"""
SuppressionRule model.

A SuppressionRule describes when to silence alerts temporarily or permanently.
It can be attached to a rule (rule_id) or be general.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
import uuid
from enum import Enum


class SuppressionScope(str, Enum):
    """
    Scope of suppression: whether suppression is per-rule, per-host, global, etc.
    """
    GLOBAL = "global"
    RULE = "rule"
    HOST = "host"
    TAG = "tag"


@dataclass
class SuppressionRule:
    """
    Data model for suppression entries or templates.

    Attributes:
        id: unique id
        name: friendly name
        selector: dict describing how to match incoming events. E.g. {"path": "host", "equals": "db-1"}
                  The matching logic is intentionally simple; for richer selectors use conditions from rules package.
        scope: one of SuppressionScope
        duration_seconds: TTL in seconds (0 or None -> indefinite until manually removed)
        created_at: timestamp
        expires_at: computed if duration_seconds provided
        reason: human-readable reason
        metadata: free-form dict for additional usage
    """
    name: str
    selector: Dict[str, Any]  # e.g., {"path": "host", "equals": "db-01"}
    scope: SuppressionScope = SuppressionScope.GLOBAL
    duration_seconds: Optional[int] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        if isinstance(self.duration_seconds, int) and self.duration_seconds > 0:
            self.expires_at = self.created_at + timedelta(seconds=int(self.duration_seconds))
        else:
            self.expires_at = None

    def is_expired(self, at_time: Optional[datetime] = None) -> bool:
        from datetime import datetime, timezone
        at_time = at_time or datetime.now(timezone.utc)
        return self.expires_at is not None and at_time >= self.expires_at

    def matches_event(self, event: Dict[str, Any]) -> bool:
        """
        Naive matching: supports selector with:
          - "path" (dotted path into event)
          - "equals" optional value to equal
          - "exists" optional bool
        More advanced selectors should use the conditions module.
        """
        path = self.selector.get("path")
        if not path:
            return True  # match everything if no path specified

        # safe nested lookup
        cur = event
        for p in path.split("."):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                if self.selector.get("exists", False):
                    return False
                return False

        if "equals" in self.selector:
            return cur == self.selector["equals"]
        if "contains" in self.selector:
            try:
                return self.selector["contains"] in cur
            except Exception:
                return False
        # default: presence is match
        return True
