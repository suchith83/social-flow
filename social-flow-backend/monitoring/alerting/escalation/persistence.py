# Persist incidents and escalation state
"""
Persistence adapter interfaces.

Provides a lightweight interface for saving incidents, escalation state and audit logs.
This is intentionally minimal — swap implementations with Redis, Postgres, DynamoDB, etc.
"""

from typing import Protocol, Dict, Any, Optional
from .incident import Incident
import threading
import logging

logger = logging.getLogger(__name__)


class PersistenceAdapter(Protocol):
    """
    Protocol for persistence adapters used by Escalator to store incident state
    and escalation progress.
    """

    def save_incident(self, incident: Incident) -> None:
        ...

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        ...

    def update_incident(self, incident: Incident) -> None:
        ...

    def log_audit(self, incident_id: str, message: str, extra: Dict[str, Any] = None) -> None:
        ...


class InMemoryPersistence:
    """
    A thread-safe in-memory persistence adapter for testing and small deployments.
    Not suitable for multi-process or production use.
    """

    def __init__(self):
        self._incidents = {}
        self._audit = {}
        self._lock = threading.RLock()

    def save_incident(self, incident: Incident) -> None:
        with self._lock:
            self._incidents[incident.id] = incident
            self._audit.setdefault(incident.id, []).append({"ts": incident.created_at.isoformat(), "event": "created"})

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        with self._lock:
            return self._incidents.get(incident_id)

    def update_incident(self, incident: Incident) -> None:
        with self._lock:
            if incident.id not in self._incidents:
                raise KeyError("Incident not found")
            self._incidents[incident.id] = incident
            self._audit.setdefault(incident.id, []).append({"ts": "now", "event": f"status:{incident.status}"})

    def log_audit(self, incident_id: str, message: str, extra: Dict[str, Any] = None) -> None:
        with self._lock:
            self._audit.setdefault(incident_id, []).append({"ts": "now", "event": message, "extra": extra})
            logger.debug("Audit: %s %s", incident_id, message)
