# On-call schedule management
"""
On-call schedule helpers.

This module provides a minimal, pluggable on-call schedule representation and
an adapter for integrating with external schedule systems (e.g., PagerDuty, Opsgenie, Google Calendar).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta


@dataclass
class OnCallEntry:
    """
    Single on-call entry.

    Attributes:
        user_id: identifier (string) for on-call user (email, username, or external id)
        contact: dict with contact info (email, phone, slack_id, etc.)
        start: start datetime (UTC)
        end: end datetime (UTC)
        metadata: extra fields
    """
    user_id: str
    contact: Dict[str, str]
    start: datetime
    end: datetime
    metadata: Dict[str, Any] = None


class OnCallSchedule:
    """
    In-memory schedule with lookup APIs.

    Note: In production you would implement an adapter that calls PagerDuty/Oncall API
    to fetch active on-call users for a given schedule. This class provides a local
    implementation and extension points for remote adapters.
    """

    def __init__(self, entries: Optional[List[OnCallEntry]] = None):
        self._entries = entries or []

    def add_entry(self, entry: OnCallEntry) -> None:
        """Add an on-call entry."""
        self._entries.append(entry)

    def get_current_oncall(self, at_time: Optional[datetime] = None) -> List[OnCallEntry]:
        """Return list of OnCallEntry active at a specific time (default now)."""
        at_time = at_time or datetime.now(timezone.utc)
        return [e for e in self._entries if e.start <= at_time < e.end]

    def get_oncall_for_window(self, start: datetime, end: datetime) -> List[OnCallEntry]:
        """Return entries overlapping a time window."""
        return [e for e in self._entries if not (e.end <= start or e.start >= end)]

    def find_by_user(self, user_id: str) -> Optional[OnCallEntry]:
        """Return the active entry for a given user id if any."""
        now = datetime.now(timezone.utc)
        for e in self.get_current_oncall(now):
            if e.user_id == user_id:
                return e
        return None
