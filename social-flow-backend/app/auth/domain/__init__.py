"""
Auth Domain Layer

Domain layer for the Auth bounded context.
Contains entities, value objects, and repository interfaces.
"""

from app.auth.domain.entities import (
    UserEntity,
    UserCreatedEvent,
    UserVerifiedEvent,
    UserBannedEvent,
    UserSuspendedEvent,
    UserEmailChangedEvent,
)
from app.auth.domain.value_objects import (
    Email,
    Username,
    Password,
    AccountStatus,
    PrivacyLevel,
    SuspensionDetails,
    BanDetails,
)
from app.auth.domain.repositories import IUserRepository

__all__ = [
    # Entities
    "UserEntity",
    # Domain events
    "UserCreatedEvent",
    "UserVerifiedEvent",
    "UserBannedEvent",
    "UserSuspendedEvent",
    "UserEmailChangedEvent",
    # Value objects
    "Email",
    "Username",
    "Password",
    "AccountStatus",
    "PrivacyLevel",
    "SuspensionDetails",
    "BanDetails",
    # Repository interfaces
    "IUserRepository",
]
