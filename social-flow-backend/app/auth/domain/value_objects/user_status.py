"""
User Status Value Objects

Represents various status states for user accounts.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


class AccountStatus(str, Enum):
    """User account status enumeration."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"
    PENDING_VERIFICATION = "pending_verification"


class PrivacyLevel(str, Enum):
    """User privacy level enumeration."""
    
    PUBLIC = "public"
    FRIENDS = "friends"
    PRIVATE = "private"


@dataclass(frozen=True)
class SuspensionDetails:
    """
    Value object representing suspension details.
    
    Immutable record of why and when a user was suspended.
    """
    
    reason: str
    suspended_at: datetime
    ends_at: Optional[datetime]
    
    def __post_init__(self) -> None:
        """Validate suspension details."""
        if not self.reason or len(self.reason.strip()) == 0:
            raise ValueError("Suspension reason cannot be empty")
        
        if self.ends_at and self.ends_at <= self.suspended_at:
            raise ValueError("Suspension end date must be after start date")
    
    @property
    def is_permanent(self) -> bool:
        """Check if suspension is permanent."""
        return self.ends_at is None
    
    @property
    def is_expired(self) -> bool:
        """Check if suspension has expired."""
        if self.is_permanent:
            return False
        return datetime.utcnow() >= self.ends_at  # type: ignore


@dataclass(frozen=True)
class BanDetails:
    """
    Value object representing ban details.
    
    Immutable record of why and when a user was banned.
    """
    
    reason: str
    banned_at: datetime
    
    def __post_init__(self) -> None:
        """Validate ban details."""
        if not self.reason or len(self.reason.strip()) == 0:
            raise ValueError("Ban reason cannot be empty")
