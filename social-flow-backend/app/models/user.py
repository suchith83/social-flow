"""
User models for authentication and user management.

This module defines the unified User model and related models for
the Social Flow backend, consolidating all user-related functionality.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum as SQLEnum,
    Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
)
from sqlalchemy.orm import Mapped, relationship

from app.models.base import CommonBase
from app.models.types import JSONB, UUID

if TYPE_CHECKING:
    from app.models.video import Video
    from app.models.post import Post
    from app.models.payment import Payment, Subscription
    from app.models.social import Follow, Like, Comment


class UserRole(str, PyEnum):
    """User role enumeration."""
    USER = "user"
    CREATOR = "creator"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class UserStatus(str, PyEnum):
    """User account status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"
    PENDING_VERIFICATION = "pending_verification"


class PrivacyLevel(str, PyEnum):
    """Privacy level for user profiles."""
    PUBLIC = "public"
    FRIENDS_ONLY = "friends"
    PRIVATE = "private"


class User(CommonBase):
    """
    Unified User model for the Social Flow platform.
    
    This model consolidates all user-related functionality including:
    - Authentication (email, phone, password)
    - Profile information (bio, avatar, etc.)
    - Verification status
    - 2FA/TOTP
    - Payment integration (Stripe)
    - Social stats (followers, views, etc.)
    - Preferences and settings
    - Moderation (ban, suspension)
    """
    
    __tablename__ = "users"
    
    # ==================== Basic Information ====================
    username = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        doc="Unique username (alphanumeric + underscore)"
    )
    
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="Primary email address"
    )
    
    phone_number = Column(
        String(20),
        unique=True,
        nullable=True,
        index=True,
        doc="Phone number with country code (E.164 format)"
    )
    
    password_hash = Column(
        String(255),
        nullable=False,
        default="$2b$12$2uZb1w1XDummyHashValueForTestsuQx8J5jH1m4eY0bQeWm1Y4ZKX7Pu",  # safe placeholder
        doc="Bcrypt hashed password"
    )

    # --------------------------------------------------
    # Backwards Compatibility
    # --------------------------------------------------
    def __init__(self, **kwargs):  # type: ignore[override]
        # Some legacy tests/fixtures still pass 'hashed_password'. Map it.
        if 'hashed_password' in kwargs and 'password_hash' not in kwargs:
            kwargs['password_hash'] = kwargs.pop('hashed_password')
        # Legacy boolean activation flag
        if 'is_active' in kwargs:
            if kwargs.pop('is_active'):
                kwargs.setdefault('status', UserStatus.ACTIVE)
        super().__init__(**kwargs)
    
    # ==================== Profile Information ====================
    display_name = Column(
        String(100),
        nullable=True,
        doc="Display name shown to other users"
    )
    
    bio = Column(
        Text,
        nullable=True,
        doc="User biography/about section"
    )
    
    avatar_url = Column(
        String(500),
        nullable=True,
        doc="Profile picture URL (S3/CloudFront)"
    )
    
    cover_image_url = Column(
        String(500),
        nullable=True,
        doc="Profile cover image URL"
    )
    
    website = Column(
        String(500),
        nullable=True,
        doc="Personal website URL"
    )
    
    location = Column(
        String(100),
        nullable=True,
        doc="User location (city, country)"
    )
    
    date_of_birth = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Date of birth for age verification"
    )
    
    # ==================== Account Status ====================
    role = Column(
        SQLEnum(UserRole),
        default=UserRole.USER,
        nullable=False,
        index=True,
        doc="User role for RBAC"
    )
    
    status = Column(
        SQLEnum(UserStatus),
        default=UserStatus.PENDING_VERIFICATION,
        nullable=False,
        index=True,
        doc="Account status"
    )
    
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Email verification status"
    )
    
    is_phone_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Phone number verification status"
    )
    
    is_creator = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Creator status (can monetize content)"
    )
    
    is_premium = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Premium subscription status"
    )
    
    # ==================== 2FA / Security ====================
    two_factor_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="2FA enabled status"
    )
    
    two_factor_secret = Column(
        String(100),
        nullable=True,
        doc="TOTP secret key (encrypted)"
    )
    
    backup_codes = Column(
        JSONB,
        default=[],
        nullable=False,
        doc="Array of backup codes for 2FA"
    )
    
    # ==================== OAuth / External Auth ====================
    google_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Google OAuth ID"
    )
    
    facebook_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Facebook OAuth ID"
    )
    
    twitter_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Twitter/X OAuth ID"
    )
    
    github_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="GitHub OAuth ID"
    )
    
    # ==================== Payment Integration ====================
    stripe_customer_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Stripe customer ID"
    )
    
    stripe_connect_account_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Stripe Connect account ID (for creators)"
    )
    
    stripe_connect_onboarded = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Stripe Connect onboarding completed"
    )
    
    # ==================== Social Stats (Denormalized for Performance) ====================
    follower_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        index=True,
        doc="Total number of followers"
    )
    
    following_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of users being followed"
    )
    
    video_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of videos uploaded"
    )
    
    post_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of posts created"
    )
    
    total_views = Column(
        BigInteger,
        default=0,
        nullable=False,
        index=True,
        doc="Total views across all content"
    )
    
    total_likes = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total likes received"
    )
    
    total_watch_time = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total watch time in seconds"
    )
    
    # ==================== Revenue Stats (Denormalized) ====================
    total_revenue = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Total revenue earned (USD)"
    )
    
    pending_payout = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Pending payout amount (USD)"
    )
    
    lifetime_payout = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Lifetime payout amount (USD)"
    )
    
    # ==================== Moderation ====================
    ban_reason = Column(
        Text,
        nullable=True,
        doc="Reason for account ban"
    )
    
    banned_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when account was banned"
    )
    
    banned_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="ID of moderator who banned the account"
    )
    
    suspension_reason = Column(
        Text,
        nullable=True,
        doc="Reason for account suspension"
    )
    
    suspended_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when account was suspended"
    )
    
    suspension_ends_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when suspension expires"
    )
    
    suspended_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="ID of moderator who suspended the account"
    )
    
    # ==================== Preferences ====================
    privacy_level = Column(
        SQLEnum(PrivacyLevel),
        default=PrivacyLevel.PUBLIC,
        nullable=False,
        doc="Profile privacy level"
    )
    
    email_notifications = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Email notifications enabled"
    )
    
    push_notifications = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Push notifications enabled"
    )
    
    marketing_emails = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Marketing emails enabled"
    )
    
    show_activity_status = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Show online/activity status to others"
    )
    
    allow_messages_from = Column(
        String(20),
        default="everyone",
        nullable=False,
        doc="Who can send messages: everyone, following, none"
    )
    
    content_language = Column(
        String(10),
        default="en",
        nullable=False,
        doc="Preferred content language (ISO 639-1)"
    )
    
    timezone = Column(
        String(50),
        default="UTC",
        nullable=False,
        doc="User timezone (IANA timezone name)"
    )
    
    # ==================== Activity Tracking ====================
    last_login_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Last login timestamp"
    )
    
    last_login_ip = Column(
        String(45),
        nullable=True,
        doc="Last login IP address (IPv6 compatible)"
    )
    
    last_active_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Last activity timestamp"
    )
    
    login_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total login count"
    )
    
    # ==================== Verification Timestamps ====================
    email_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Email verification timestamp"
    )
    
    phone_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Phone verification timestamp"
    )
    
    creator_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Creator verification timestamp"
    )
    
    # ==================== Relationships ====================
    # Content relationships
    videos: Mapped[list["Video"]] = relationship(
        "Video",
        back_populates="owner",
        foreign_keys="Video.owner_id",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    posts: Mapped[list["Post"]] = relationship(
        "Post",
        back_populates="owner",
        foreign_keys="Post.owner_id",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Social relationships
    followers: Mapped[list["Follow"]] = relationship(
        "Follow",
        foreign_keys="Follow.following_id",
        back_populates="following_user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    following: Mapped[list["Follow"]] = relationship(
        "Follow",
        foreign_keys="Follow.follower_id",
        back_populates="follower_user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    comments: Mapped[list["Comment"]] = relationship(
        "Comment",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    likes: Mapped[list["Like"]] = relationship(
        "Like",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Payment relationships
    payments: Mapped[list["Payment"]] = relationship(
        "Payment",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    subscriptions: Mapped[list["Subscription"]] = relationship(
        "Subscription",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Moderation relationships (self-referential)
    banned_users = relationship(
        "User",
        foreign_keys=[banned_by_id],
        remote_side="User.id",
        backref="banned_by_user"
    )
    
    suspended_users = relationship(
        "User",
        foreign_keys=[suspended_by_id],
        remote_side="User.id",
        backref="suspended_by_user"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        # Composite indexes for common query patterns
        Index('idx_user_status_role', 'status', 'role'),
        Index('idx_user_creator_verified', 'is_creator', 'is_verified'),
        Index('idx_user_follower_count', 'follower_count'),
        Index('idx_user_total_views', 'total_views'),
        Index('idx_user_last_active', 'last_active_at'),
        Index('idx_user_created_at', 'created_at'),
        
        # Unique constraints for OAuth IDs
        UniqueConstraint('google_id', name='uq_user_google_id'),
        UniqueConstraint('facebook_id', name='uq_user_facebook_id'),
        UniqueConstraint('twitter_id', name='uq_user_twitter_id'),
        UniqueConstraint('github_id', name='uq_user_github_id'),
        
        # Check constraints
        {'postgresql_partition_by': 'RANGE (created_at)'},  # Prepare for partitioning
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
    
    @property
    def is_private(self) -> bool:
        """Check if user profile is private."""
        return self.privacy_level == PrivacyLevel.PRIVATE
    
    @property
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE

    @is_active.setter
    def is_active(self, value: bool):  # type: ignore[override]
        # Legacy tests set is_active directly; translate to status.
        if value:
            self.status = UserStatus.ACTIVE
        else:
            if self.status == UserStatus.ACTIVE:
                self.status = UserStatus.INACTIVE
    
    def is_banned(self) -> bool:
        """Check if user is currently banned."""
        return self.status == UserStatus.BANNED
    
    def is_suspended(self) -> bool:
        """Check if user is currently suspended."""
        if self.status != UserStatus.SUSPENDED:
            return False
        
        # Check if suspension has expired
        if self.suspension_ends_at and self.suspension_ends_at < datetime.utcnow():
            return False
            
        return True
    
    def can_post_content(self) -> bool:
        """Check if user can post content."""
        return (
            self.status == UserStatus.ACTIVE
            and not self.is_banned()
            and not self.is_suspended()
            and not self.is_deleted
        )
    
    def can_monetize(self) -> bool:
        """Check if user can monetize content."""
        return (
            self.is_creator
            and self.stripe_connect_onboarded
            and self.can_post_content()
        )

    # Compatibility / convenience properties
    @property
    def is_superuser(self) -> bool:  # noqa: D401
        """Compatibility property expected by routes; true if user has an admin-level role.

        The codebase historically referenced `is_superuser` while the consolidated
        user model only retained a `role` enum. We treat ADMIN and SUPER_ADMIN
        as superuser roles. This avoids attribute errors and centralizes the
        permission logic in one place.
        """
        try:
            return self.role in {UserRole.ADMIN, UserRole.SUPER_ADMIN}
        except Exception:
            return False


class EmailVerificationToken(CommonBase):
    """Email verification tokens."""
    
    __tablename__ = "email_verification_tokens"
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="Verification token"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Token expiration timestamp"
    )
    
    used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when token was used"
    )
    
    # Relationship
    user = relationship("User", backref="verification_tokens")
    
    def is_valid(self) -> bool:
        """Check if token is valid."""
        return (
            self.used_at is None
            and self.expires_at > datetime.utcnow()
            and not self.is_deleted
        )


class PasswordResetToken(CommonBase):
    """Password reset tokens."""
    
    __tablename__ = "password_reset_tokens"
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="Reset token"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Token expiration timestamp"
    )
    
    used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when token was used"
    )
    
    # Relationship
    user = relationship("User", backref="password_reset_tokens")
    
    def is_valid(self) -> bool:
        """Check if token is valid."""
        return (
            self.used_at is None
            and self.expires_at > datetime.utcnow()
            and not self.is_deleted
        )


# Export models
__all__ = [
    'User',
    'UserRole',
    'UserStatus',
    'PrivacyLevel',
    'EmailVerificationToken',
    'PasswordResetToken',
]
