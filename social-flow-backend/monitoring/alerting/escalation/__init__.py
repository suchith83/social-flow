# Package initializer for escalation module
"""
Escalation package for monitoring alerts.

Exports:
- Incident
- EscalationPolicy
- OnCallSchedule
- Escalator and RoundRobinEscalator
- retry/backoff utilities
- persistence interfaces
"""

from .incident import Incident, IncidentStatus
from .escalation_policy import EscalationPolicy, EscalationLevel
from .oncall_schedule import OnCallSchedule, OnCallEntry
from .escalator import Escalator
from .round_robin_escalator import RoundRobinEscalator
from .retry_backoff import RetryBackoff
from .persistence import PersistenceAdapter, InMemoryPersistence

__all__ = [
    "Incident",
    "IncidentStatus",
    "EscalationPolicy",
    "EscalationLevel",
    "OnCallSchedule",
    "OnCallEntry",
    "Escalator",
    "RoundRobinEscalator",
    "RetryBackoff",
    "PersistenceAdapter",
    "InMemoryPersistence",
]
