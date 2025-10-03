"""
Shared Domain Layer - Shared Kernel

Contains domain concepts shared across all bounded contexts.
"""

from app.shared.domain.base import (
    BaseEntity,
    AggregateRoot,
    DomainEvent,
)
from app.shared.domain.value_objects import (
    UserRole,
    VideoStatus,
    VideoVisibility,
    PostVisibility,
)

__all__ = [
    # Base classes
    "BaseEntity",
    "AggregateRoot",
    "DomainEvent",
    # Shared value objects
    "UserRole",
    "VideoStatus",
    "VideoVisibility",
    "PostVisibility",
]
