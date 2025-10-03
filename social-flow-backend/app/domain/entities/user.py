"""
User Domain Entity

Rich domain model for User with business logic and invariants.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from app.domain.entities.base import AggregateRoot, DomainEvent
from app.domain.value_objects import Email, Username, UserRole


class UserCreatedEvent(DomainEvent):
    """Event raised when a user is created."""
    
    def __init__(self, user_id: UUID, username: str, email: str):
        super().__init__(
            event_type="user.created",
            data={
                "user_id": str(user_id),
                "username": username,
                "email": email,
            }
        )


class UserVerifiedEvent(DomainEvent):
    """Event raised when a user is verified."""
    
    def __init__(self, user_id: UUID):
        super().__init__(
            event_type="user.verified",
            data={"user_id": str(user_id)}
        )


class UserBannedEvent(DomainEvent):
    """Event raised when a user is banned."""
    
    def __init__(self, user_id: UUID, reason: str):
        super().__init__(
            event_type="user.banned",
            data={
                "user_id": str(user_id),
                "reason": reason,
            }
        )


class UserEntity(AggregateRoot):
    """
    User domain entity with business logic.
    
    Encapsulates user-related business rules and invariants.
    """
    
    def __init__(
        self,
        username: Username,
        email: Email,
        password_hash: str,
        display_name: str,
        id: Optional[UUID] = None,
        role: UserRole = UserRole.USER,
    ):
        super().__init__(id)
        self._username = username
        self._email = email
        self._password_hash = password_hash
        self._display_name = display_name
        self._role = role
        
        # Profile information
        self._bio: Optional[str] = None
        self._avatar_url: Optional[str] = None
        self._website: Optional[str] = None
        self._location: Optional[str] = None
        
        # Account status
        self._is_active = True
        self._is_verified = False
        self._is_banned = False
        self._is_suspended = False
        
        # Ban/suspension details
        self._ban_reason: Optional[str] = None
        self._banned_at: Optional[datetime] = None
        self._suspension_reason: Optional[str] = None
        self._suspended_at: Optional[datetime] = None
        self._suspension_ends_at: Optional[datetime] = None
        
        # Social metrics
        self._followers_count = 0
        self._following_count = 0
        self._posts_count = 0
        self._videos_count = 0
        self._total_views = 0
        self._total_likes = 0
        
        # Preferences
        self._email_notifications = True
        self._push_notifications = True
        self._privacy_level = "public"
        
        # Last login
        self._last_login_at: Optional[datetime] = None
        
        # Raise creation event
        self._raise_event(UserCreatedEvent(self.id, str(username), str(email)))
    
    # Properties (getters)
    
    @property
    def username(self) -> Username:
        return self._username
    
    @property
    def email(self) -> Email:
        return self._email
    
    @property
    def password_hash(self) -> str:
        return self._password_hash
    
    @property
    def display_name(self) -> str:
        return self._display_name
    
    @property
    def role(self) -> UserRole:
        return self._role
    
    @property
    def bio(self) -> Optional[str]:
        return self._bio
    
    @property
    def avatar_url(self) -> Optional[str]:
        return self._avatar_url
    
    @property
    def website(self) -> Optional[str]:
        return self._website
    
    @property
    def location(self) -> Optional[str]:
        return self._location
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    @property
    def is_verified(self) -> bool:
        return self._is_verified
    
    @property
    def is_banned(self) -> bool:
        return self._is_banned
    
    @property
    def is_suspended(self) -> bool:
        return self._is_suspended
    
    @property
    def followers_count(self) -> int:
        return self._followers_count
    
    @property
    def following_count(self) -> int:
        return self._following_count
    
    @property
    def posts_count(self) -> int:
        return self._posts_count
    
    @property
    def videos_count(self) -> int:
        return self._videos_count
    
    @property
    def total_views(self) -> int:
        return self._total_views
    
    @property
    def total_likes(self) -> int:
        return self._total_likes
    
    @property
    def last_login_at(self) -> Optional[datetime]:
        return self._last_login_at
    
    # Business methods
    
    def can_post(self) -> bool:
        """Check if user can create posts."""
        return (
            self._is_active
            and not self._is_banned
            and not self._is_suspended
            and self._is_verified
        )
    
    def can_comment(self) -> bool:
        """Check if user can comment."""
        return (
            self._is_active
            and not self._is_banned
            and not self._is_suspended
        )
    
    def can_upload_video(self) -> bool:
        """Check if user can upload videos."""
        return (
            self._is_active
            and not self._is_banned
            and not self._is_suspended
            and self._is_verified
            and self._role in [UserRole.CREATOR, UserRole.ADMIN, UserRole.SUPER_ADMIN]
        )
    
    def can_moderate(self) -> bool:
        """Check if user has moderation privileges."""
        return self._role in [UserRole.MODERATOR, UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def can_administrate(self) -> bool:
        """Check if user has admin privileges."""
        return self._role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def is_account_active(self) -> bool:
        """Check if account is fully active."""
        if self._is_banned or not self._is_active:
            return False
        
        if self._is_suspended:
            # Check if suspension has expired
            if self._suspension_ends_at and datetime.utcnow() > self._suspension_ends_at:
                self._lift_suspension()
                return True
            return False
        
        return True
    
    # Mutation methods
    
    def update_profile(
        self,
        display_name: Optional[str] = None,
        bio: Optional[str] = None,
        website: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        """Update user profile information."""
        if not self.is_account_active():
            raise ValueError("Cannot update profile: account is not active")
        
        if display_name is not None:
            if len(display_name) < 1 or len(display_name) > 100:
                raise ValueError("Display name must be 1-100 characters")
            self._display_name = display_name
        
        if bio is not None:
            if len(bio) > 500:
                raise ValueError("Bio must be max 500 characters")
            self._bio = bio
        
        if website is not None:
            self._website = website
        
        if location is not None:
            self._location = location
        
        self._mark_updated()
        self._increment_version()
    
    def update_avatar(self, avatar_url: str) -> None:
        """Update user avatar."""
        if not self.is_account_active():
            raise ValueError("Cannot update avatar: account is not active")
        
        self._avatar_url = avatar_url
        self._mark_updated()
        self._increment_version()
    
    def change_email(self, new_email: Email) -> None:
        """Change user email (requires re-verification)."""
        if not self.is_account_active():
            raise ValueError("Cannot change email: account is not active")
        
        self._email = new_email
        self._is_verified = False  # Require re-verification
        self._mark_updated()
        self._increment_version()
    
    def change_password(self, new_password_hash: str) -> None:
        """Change user password."""
        if not self.is_account_active():
            raise ValueError("Cannot change password: account is not active")
        
        self._password_hash = new_password_hash
        self._mark_updated()
        self._increment_version()
    
    def verify_account(self) -> None:
        """Verify user account."""
        if self._is_verified:
            return
        
        self._is_verified = True
        self._mark_updated()
        self._increment_version()
        self._raise_event(UserVerifiedEvent(self.id))
    
    def ban(self, reason: str) -> None:
        """Ban user account."""
        if self._is_banned:
            return
        
        self._is_banned = True
        self._is_active = False
        self._ban_reason = reason
        self._banned_at = datetime.utcnow()
        self._mark_updated()
        self._increment_version()
        self._raise_event(UserBannedEvent(self.id, reason))
    
    def unban(self) -> None:
        """Unban user account."""
        if not self._is_banned:
            return
        
        self._is_banned = False
        self._is_active = True
        self._ban_reason = None
        self._banned_at = None
        self._mark_updated()
        self._increment_version()
    
    def suspend(self, reason: str, ends_at: datetime) -> None:
        """Suspend user account temporarily."""
        if self._is_banned:
            raise ValueError("Cannot suspend: user is banned")
        
        if ends_at <= datetime.utcnow():
            raise ValueError("Suspension end date must be in the future")
        
        self._is_suspended = True
        self._suspension_reason = reason
        self._suspended_at = datetime.utcnow()
        self._suspension_ends_at = ends_at
        self._mark_updated()
        self._increment_version()
    
    def _lift_suspension(self) -> None:
        """Lift suspension (internal use)."""
        self._is_suspended = False
        self._suspension_reason = None
        self._suspended_at = None
        self._suspension_ends_at = None
        self._mark_updated()
        self._increment_version()
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self._is_active = False
        self._mark_updated()
        self._increment_version()
    
    def reactivate(self) -> None:
        """Reactivate user account."""
        if self._is_banned:
            raise ValueError("Cannot reactivate: user is banned")
        
        self._is_active = True
        self._mark_updated()
        self._increment_version()
    
    def promote_role(self, new_role: UserRole) -> None:
        """Promote user to a new role."""
        if new_role.value <= self._role.value:
            raise ValueError("New role must be higher than current role")
        
        self._role = new_role
        self._mark_updated()
        self._increment_version()
    
    def record_login(self) -> None:
        """Record user login."""
        self._last_login_at = datetime.utcnow()
        self._mark_updated()
    
    def increment_followers(self) -> None:
        """Increment followers count."""
        self._followers_count += 1
        self._mark_updated()
    
    def decrement_followers(self) -> None:
        """Decrement followers count."""
        if self._followers_count > 0:
            self._followers_count -= 1
            self._mark_updated()
    
    def increment_following(self) -> None:
        """Increment following count."""
        self._following_count += 1
        self._mark_updated()
    
    def decrement_following(self) -> None:
        """Decrement following count."""
        if self._following_count > 0:
            self._following_count -= 1
            self._mark_updated()
    
    def increment_posts(self) -> None:
        """Increment posts count."""
        self._posts_count += 1
        self._mark_updated()
    
    def increment_videos(self) -> None:
        """Increment videos count."""
        self._videos_count += 1
        self._mark_updated()
    
    def add_views(self, count: int) -> None:
        """Add to total views."""
        if count < 0:
            raise ValueError("View count cannot be negative")
        self._total_views += count
        self._mark_updated()
    
    def add_likes(self, count: int) -> None:
        """Add to total likes."""
        if count < 0:
            raise ValueError("Like count cannot be negative")
        self._total_likes += count
        self._mark_updated()
