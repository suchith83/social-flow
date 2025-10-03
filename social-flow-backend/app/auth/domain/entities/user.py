"""
User Domain Entity - Auth Bounded Context

Rich domain model for User with business logic and invariants.
Uses value objects for type safety and validation.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from app.shared.domain.base import AggregateRoot, DomainEvent
from app.shared.domain.value_objects import UserRole
from app.auth.domain.value_objects import (
    Email,
    Username,
    Password,
    AccountStatus,
    PrivacyLevel,
    SuspensionDetails,
    BanDetails,
)


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


class UserSuspendedEvent(DomainEvent):
    """Event raised when a user is suspended."""
    
    def __init__(self, user_id: UUID, reason: str, ends_at: Optional[datetime]):
        super().__init__(
            event_type="user.suspended",
            data={
                "user_id": str(user_id),
                "reason": reason,
                "ends_at": ends_at.isoformat() if ends_at else None,
            }
        )


class UserEmailChangedEvent(DomainEvent):
    """Event raised when a user changes their email."""
    
    def __init__(self, user_id: UUID, old_email: str, new_email: str):
        super().__init__(
            event_type="user.email_changed",
            data={
                "user_id": str(user_id),
                "old_email": old_email,
                "new_email": new_email,
            }
        )


class UserEntity(AggregateRoot):
    """
    User domain entity with business logic.
    
    Encapsulates user-related business rules and invariants.
    Uses value objects for validated domain primitives.
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
        
        # Account status using value objects
        self._account_status = AccountStatus.PENDING_VERIFICATION
        self._privacy_level = PrivacyLevel.PUBLIC
        self._suspension_details: Optional[SuspensionDetails] = None
        self._ban_details: Optional[BanDetails] = None
        
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
        
        # Last login
        self._last_login_at: Optional[datetime] = None
        
        # Raise creation event
        self._raise_event(UserCreatedEvent(self.id, username.value, email.value))
    
    # Properties (getters)
    
    @property
    def username(self) -> Username:
        """Get username value object."""
        return self._username
    
    @property
    def email(self) -> Email:
        """Get email value object."""
        return self._email
    
    @property
    def password_hash(self) -> str:
        """Get password hash (never expose raw password)."""
        return self._password_hash
    
    @property
    def display_name(self) -> str:
        """Get display name."""
        return self._display_name
    
    @property
    def role(self) -> UserRole:
        """Get user role."""
        return self._role
    
    @property
    def bio(self) -> Optional[str]:
        """Get user bio."""
        return self._bio
    
    @property
    def avatar_url(self) -> Optional[str]:
        """Get avatar URL."""
        return self._avatar_url
    
    @property
    def website(self) -> Optional[str]:
        """Get website URL."""
        return self._website
    
    @property
    def location(self) -> Optional[str]:
        """Get user location."""
        return self._location
    
    @property
    def account_status(self) -> AccountStatus:
        """Get account status."""
        return self._account_status
    
    @property
    def privacy_level(self) -> PrivacyLevel:
        """Get privacy level."""
        return self._privacy_level
    
    @property
    def suspension_details(self) -> Optional[SuspensionDetails]:
        """Get suspension details if suspended."""
        return self._suspension_details
    
    @property
    def ban_details(self) -> Optional[BanDetails]:
        """Get ban details if banned."""
        return self._ban_details
    
    @property
    def is_active(self) -> bool:
        """Check if account is active."""
        return self._account_status == AccountStatus.ACTIVE
    
    @property
    def is_verified(self) -> bool:
        """Check if account is verified."""
        return self._account_status != AccountStatus.PENDING_VERIFICATION
    
    @property
    def is_banned(self) -> bool:
        """Check if account is banned."""
        return self._account_status == AccountStatus.BANNED
    
    @property
    def is_suspended(self) -> bool:
        """Check if account is suspended."""
        return self._account_status == AccountStatus.SUSPENDED
    
    @property
    def is_inactive(self) -> bool:
        """Check if account is inactive."""
        return self._account_status == AccountStatus.INACTIVE
    
    @property
    def followers_count(self) -> int:
        """Get followers count."""
        return self._followers_count
    
    @property
    def following_count(self) -> int:
        """Get following count."""
        return self._following_count
    
    @property
    def posts_count(self) -> int:
        """Get posts count."""
        return self._posts_count
    
    @property
    def videos_count(self) -> int:
        """Get videos count."""
        return self._videos_count
    
    @property
    def total_views(self) -> int:
        """Get total views."""
        return self._total_views
    
    @property
    def total_likes(self) -> int:
        """Get total likes."""
        return self._total_likes
    
    @property
    def last_login_at(self) -> Optional[datetime]:
        """Get last login timestamp."""
        return self._last_login_at
    
    @property
    def email_notifications(self) -> bool:
        """Check if email notifications are enabled."""
        return self._email_notifications
    
    @property
    def push_notifications(self) -> bool:
        """Check if push notifications are enabled."""
        return self._push_notifications
    
    # Business logic methods
    
    def can_post(self) -> bool:
        """Check if user can create posts."""
        return (
            self._account_status == AccountStatus.ACTIVE
            and self.is_verified
        )
    
    def can_comment(self) -> bool:
        """Check if user can comment."""
        return self._account_status == AccountStatus.ACTIVE
    
    def can_upload_video(self) -> bool:
        """Check if user can upload videos."""
        return (
            self._account_status == AccountStatus.ACTIVE
            and self.is_verified
            and self._role in [UserRole.CREATOR, UserRole.ADMIN, UserRole.SUPER_ADMIN]
        )
    
    def can_moderate(self) -> bool:
        """Check if user has moderation privileges."""
        return self._role in [UserRole.MODERATOR, UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def can_administrate(self) -> bool:
        """Check if user has admin privileges."""
        return self._role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def is_account_active(self) -> bool:
        """
        Check if account is fully active and usable.
        
        Handles automatic suspension expiration.
        """
        if self._account_status == AccountStatus.BANNED:
            return False
        
        if self._account_status == AccountStatus.INACTIVE:
            return False
        
        if self._account_status == AccountStatus.SUSPENDED:
            # Check if suspension has expired
            if self._suspension_details and self._suspension_details.is_expired:
                self._lift_suspension()
                return True
            return False
        
        return self._account_status == AccountStatus.ACTIVE
    
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
            if website and len(website) > 500:
                raise ValueError("Website URL must be max 500 characters")
            self._website = website
        
        if location is not None:
            if location and len(location) > 100:
                raise ValueError("Location must be max 100 characters")
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
    
    def update_privacy_level(self, privacy_level: PrivacyLevel) -> None:
        """Update privacy level."""
        if not self.is_account_active():
            raise ValueError("Cannot update privacy: account is not active")
        
        self._privacy_level = privacy_level
        self._mark_updated()
        self._increment_version()
    
    def update_notification_preferences(
        self,
        email_notifications: Optional[bool] = None,
        push_notifications: Optional[bool] = None,
    ) -> None:
        """Update notification preferences."""
        if email_notifications is not None:
            self._email_notifications = email_notifications
        
        if push_notifications is not None:
            self._push_notifications = push_notifications
        
        self._mark_updated()
        self._increment_version()
    
    def change_email(self, new_email: Email) -> None:
        """
        Change user email (requires re-verification).
        
        Raises UserEmailChangedEvent.
        """
        if not self.is_account_active():
            raise ValueError("Cannot change email: account is not active")
        
        old_email = self._email.value
        self._email = new_email
        self._account_status = AccountStatus.PENDING_VERIFICATION  # Require re-verification
        self._mark_updated()
        self._increment_version()
        self._raise_event(UserEmailChangedEvent(self.id, old_email, new_email.value))
    
    def change_password(self, new_password_hash: str) -> None:
        """Change user password."""
        if not self.is_account_active():
            raise ValueError("Cannot change password: account is not active")
        
        if not new_password_hash:
            raise ValueError("Password hash cannot be empty")
        
        self._password_hash = new_password_hash
        self._mark_updated()
        self._increment_version()
    
    def verify_account(self) -> None:
        """
        Verify user account.
        
        Raises UserVerifiedEvent.
        """
        if self._account_status != AccountStatus.PENDING_VERIFICATION:
            return
        
        self._account_status = AccountStatus.ACTIVE
        self._mark_updated()
        self._increment_version()
        self._raise_event(UserVerifiedEvent(self.id))
    
    def ban(self, reason: str) -> None:
        """
        Ban user account permanently.
        
        Raises UserBannedEvent.
        """
        if self._account_status == AccountStatus.BANNED:
            return
        
        if not reason or len(reason.strip()) < 3:
            raise ValueError("Ban reason must be at least 3 characters")
        
        ban_details = BanDetails(
            reason=reason.strip(),
            banned_at=datetime.utcnow(),
        )
        
        self._account_status = AccountStatus.BANNED
        self._ban_details = ban_details
        self._suspension_details = None  # Clear any suspension
        self._mark_updated()
        self._increment_version()
        self._raise_event(UserBannedEvent(self.id, reason))
    
    def unban(self) -> None:
        """Unban user account."""
        if self._account_status != AccountStatus.BANNED:
            return
        
        self._account_status = AccountStatus.ACTIVE
        self._ban_details = None
        self._mark_updated()
        self._increment_version()
    
    def suspend(self, reason: str, ends_at: Optional[datetime] = None) -> None:
        """
        Suspend user account temporarily or permanently.
        
        Args:
            reason: Reason for suspension
            ends_at: When suspension ends (None = permanent)
        
        Raises UserSuspendedEvent.
        """
        if self._account_status == AccountStatus.BANNED:
            raise ValueError("Cannot suspend: user is banned")
        
        if not reason or len(reason.strip()) < 3:
            raise ValueError("Suspension reason must be at least 3 characters")
        
        if ends_at and ends_at <= datetime.utcnow():
            raise ValueError("Suspension end date must be in the future")
        
        suspension_details = SuspensionDetails(
            reason=reason.strip(),
            suspended_at=datetime.utcnow(),
            ends_at=ends_at,
        )
        
        self._account_status = AccountStatus.SUSPENDED
        self._suspension_details = suspension_details
        self._mark_updated()
        self._increment_version()
        self._raise_event(UserSuspendedEvent(self.id, reason, ends_at))
    
    def _lift_suspension(self) -> None:
        """Lift suspension (internal use for automatic expiration)."""
        if self._account_status != AccountStatus.SUSPENDED:
            return
        
        self._account_status = AccountStatus.ACTIVE
        self._suspension_details = None
        self._mark_updated()
        self._increment_version()
    
    def lift_suspension(self) -> None:
        """Lift suspension manually (admin action)."""
        self._lift_suspension()
    
    def deactivate(self) -> None:
        """Deactivate user account (user-initiated)."""
        if self._account_status == AccountStatus.BANNED:
            raise ValueError("Cannot deactivate: user is banned")
        
        self._account_status = AccountStatus.INACTIVE
        self._mark_updated()
        self._increment_version()
    
    def reactivate(self) -> None:
        """Reactivate user account."""
        if self._account_status == AccountStatus.BANNED:
            raise ValueError("Cannot reactivate: user is banned")
        
        self._account_status = AccountStatus.ACTIVE
        self._mark_updated()
        self._increment_version()
    
    def promote_role(self, new_role: UserRole) -> None:
        """Promote user to a new role."""
        if new_role.value <= self._role.value:
            raise ValueError("New role must be higher than current role")
        
        self._role = new_role
        self._mark_updated()
        self._increment_version()
    
    def demote_role(self, new_role: UserRole) -> None:
        """Demote user to a lower role."""
        if new_role.value >= self._role.value:
            raise ValueError("New role must be lower than current role")
        
        self._role = new_role
        self._mark_updated()
        self._increment_version()
    
    def record_login(self) -> None:
        """Record user login."""
        self._last_login_at = datetime.utcnow()
        self._mark_updated()
    
    # Social metric methods
    
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
    
    def decrement_posts(self) -> None:
        """Decrement posts count."""
        if self._posts_count > 0:
            self._posts_count -= 1
            self._mark_updated()
    
    def increment_videos(self) -> None:
        """Increment videos count."""
        self._videos_count += 1
        self._mark_updated()
    
    def decrement_videos(self) -> None:
        """Decrement videos count."""
        if self._videos_count > 0:
            self._videos_count -= 1
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
    
    def remove_likes(self, count: int) -> None:
        """Remove from total likes."""
        if count < 0:
            raise ValueError("Like count cannot be negative")
        self._total_likes = max(0, self._total_likes - count)
        self._mark_updated()
