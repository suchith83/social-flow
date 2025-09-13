# Defines incident data models and handling
"""
Incident model and related helpers.

An Incident represents a triggered alert that may be escalated according
to policies and on-call schedules.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
import uuid


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class Incident:
    """
    Incident data model.

    Attributes:
        id: unique incident id (uuid4)
        title: short summary
        message: long-form message
        created_at: timestamp
        status: IncidentStatus
        metadata: free-form dict for extra context (source, metrics, host, etc.)
        history: list of status change tuples (timestamp, status, note)
    """
    title: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: IncidentStatus = IncidentStatus.OPEN
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize incident to dict for persistence or transport."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "history": self.history,
        }

    def add_history(self, status: IncidentStatus, note: Optional[str] = None) -> None:
        """Append a status change to incident history."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status.value,
            "note": note,
        }
        self.history.append(entry)
        self.status = status
