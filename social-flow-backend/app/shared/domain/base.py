"""
Base Domain Classes - Shared Kernel

Provides common functionality for all domain entities across bounded contexts.
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class DomainEvent:
    """Base class for domain events."""
    
    event_type: str
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: Optional[UUID] = None
    data: Dict[str, Any] = field(default_factory=dict)


class BaseEntity(ABC):
    """
    Base class for all domain entities.
    
    Entities have identity and lifecycle. They are defined by their ID,
    not their attributes.
    """
    
    def __init__(self, id: Optional[UUID] = None):
        self._id = id or uuid4()
        self._domain_events: List[DomainEvent] = []
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()
    
    @property
    def id(self) -> UUID:
        """Get entity ID."""
        return self._id
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def domain_events(self) -> List[DomainEvent]:
        """Get domain events raised by this entity."""
        return self._domain_events.copy()
    
    def clear_events(self) -> None:
        """Clear all domain events."""
        self._domain_events.clear()
    
    def _raise_event(self, event: DomainEvent) -> None:
        """Raise a domain event."""
        event.aggregate_id = self._id
        self._domain_events.append(event)
    
    def _mark_updated(self) -> None:
        """Mark entity as updated."""
        self._updated_at = datetime.utcnow()
    
    def __eq__(self, other: object) -> bool:
        """Entities are equal if they have the same ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self._id)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self._id})>"


class AggregateRoot(BaseEntity):
    """
    Base class for aggregate roots.
    
    Aggregate roots are the main entities that control access to
    other entities within the aggregate boundary.
    """
    
    def __init__(self, id: Optional[UUID] = None):
        super().__init__(id)
        self._version: int = 1
    
    @property
    def version(self) -> int:
        """Get aggregate version for optimistic locking."""
        return self._version
    
    def _increment_version(self) -> None:
        """Increment version for optimistic locking."""
        self._version += 1
        self._mark_updated()
