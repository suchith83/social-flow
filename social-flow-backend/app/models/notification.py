"""  
Notification models for multi-channel messaging.

This module defines comprehensive notification models including:
- Notifications (in-app, email, push, SMS)
- User notification preferences
- Push notification tokens (FCM)
- Notification templates
"""

from __future__ import annotations

from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum,
    ForeignKey, Index, Integer, String, Text
)
from sqlalchemy.orm import Mapped, relationship

from app.models.base import CommonBase
from app.models.types import ARRAY, JSONB, UUID

if TYPE_CHECKING:
    from app.models.user import User


class NotificationType(str, PyEnum):
    """Notification type."""
    # Social interactions
    FOLLOW = "follow"
    LIKE = "like"
    COMMENT = "comment"
    MENTION = "mention"
    REPOST = "repost"
    
    # Video interactions
    VIDEO_LIKE = "video_like"
    VIDEO_COMMENT = "video_comment"
    VIDEO_UPLOADED = "video_uploaded"
    VIDEO_PROCESSED = "video_processed"
    VIDEO_MODERATION = "video_moderation"
    
    # Live streaming
    LIVE_STREAM_STARTED = "live_stream_started"
    LIVE_STREAM_ENDING = "live_stream_ending"
    STREAM_DONATION = "stream_donation"
    
    # Payments
    PAYMENT_RECEIVED = "payment_received"
    PAYMENT_FAILED = "payment_failed"
    PAYOUT_PROCESSED = "payout_processed"
    SUBSCRIPTION_STARTED = "subscription_started"
    SUBSCRIPTION_ENDING = "subscription_ending"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    
    # Moderation
    CONTENT_FLAGGED = "content_flagged"
    CONTENT_REMOVED = "content_removed"
    ACCOUNT_WARNING = "account_warning"
    ACCOUNT_SUSPENDED = "account_suspended"
    ACCOUNT_BANNED = "account_banned"
    
    # System
    SYSTEM_ANNOUNCEMENT = "system_announcement"
    SECURITY_ALERT = "security_alert"
    FEATURE_UPDATE = "feature_update"


class NotificationChannel(str, PyEnum):
    """Notification delivery channel."""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"


class NotificationStatus(str, PyEnum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class PushPlatform(str, PyEnum):
    """Push notification platform."""
    FCM = "fcm"  # Firebase Cloud Messaging
    APNS = "apns"  # Apple Push Notification Service
    WEB = "web"  # Web push


class Notification(CommonBase):
    """
    Notification model.
    
    Multi-channel notification system for in-app, email, push, and SMS.
    """
    
    __tablename__ = "notifications"
    
    # ==================== Recipient ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Recipient user ID"
    )
    
    # ==================== Notification Type ====================
    type = Column(
        SQLEnum(NotificationType),
        nullable=False,
        index=True,
        doc="Notification type"
    )
    
    # ==================== Content ====================
    title = Column(
        String(200),
        nullable=False,
        doc="Notification title"
    )
    
    body = Column(
        Text,
        nullable=False,
        doc="Notification body text"
    )
    
    image_url = Column(
        String(500),
        nullable=True,
        doc="Optional image URL"
    )
    
    # ==================== Action Data ====================
    action_url = Column(
        String(500),
        nullable=True,
        doc="Action URL (deep link)"
    )
    
    data = Column(
        JSONB,
        default={},
        nullable=False,
        doc="Additional structured data"
    )
    
    # ==================== Actor ====================
    actor_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="User who triggered the notification (optional)"
    )
    
    # ==================== Related Entities ====================
    video_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="Related video ID"
    )
    
    post_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="Related post ID"
    )
    
    comment_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="Related comment ID"
    )
    
    livestream_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="Related livestream ID"
    )
    
    payment_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        doc="Related payment ID"
    )
    
    # ==================== Delivery Channels ====================
    channels = Column(
        ARRAY(SQLEnum(NotificationChannel)),
        default=[NotificationChannel.IN_APP],
        nullable=False,
        doc="Delivery channels"
    )
    
    # ==================== Status ====================
    status = Column(
        SQLEnum(NotificationStatus),
        default=NotificationStatus.PENDING,
        nullable=False,
        index=True,
        doc="Delivery status"
    )
    
    # ==================== Timestamps ====================
    sent_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Sent timestamp"
    )
    
    delivered_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Delivery confirmation timestamp"
    )
    
    read_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Read timestamp"
    )
    
    clicked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Click timestamp"
    )
    
    # ==================== Priority ====================
    priority = Column(
        String(20),
        default="normal",
        nullable=False,
        doc="Priority (low, normal, high, urgent)"
    )
    
    # ==================== Grouping ====================
    group_key = Column(
        String(100),
        nullable=True,
        index=True,
        doc="Group key for collapsing similar notifications"
    )
    
    # ==================== Expiration ====================
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Expiration timestamp"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        foreign_keys=[user_id],
        backref="notifications"
    )
    
    actor: Mapped["User"] = relationship(
        "User",
        foreign_keys=[actor_id]
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_notification_user_created', 'user_id', 'created_at'),
        Index('idx_notification_user_status', 'user_id', 'status'),
        Index('idx_notification_user_read', 'user_id', 'read_at'),
        Index('idx_notification_type_created', 'type', 'created_at'),
        Index('idx_notification_group', 'user_id', 'group_key'),
        Index('idx_notification_expires', 'expires_at'),
        Index('idx_notification_actor', 'actor_id', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Notification(id={self.id}, user_id={self.user_id}, type={self.type})>"
    
    def mark_as_read(self) -> None:
        """Mark notification as read."""
        from datetime import datetime, timezone as tz
        if not self.read_at:
            self.read_at = datetime.now(tz.utc)
            self.status = NotificationStatus.READ
    
    def is_read(self) -> bool:
        """Check if notification has been read."""
        return self.read_at is not None
    
    def is_expired(self) -> bool:
        """Check if notification has expired."""
        from datetime import datetime, timezone as tz
        if not self.expires_at:
            return False
        return datetime.now(tz.utc) > self.expires_at


class NotificationSettings(CommonBase):
    """
    User notification preferences.
    
    Fine-grained control over notification delivery per type and channel.
    """
    
    __tablename__ = "notification_settings"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        unique=True,
        index=True,
        doc="User ID"
    )
    
    # ==================== Social Notifications ====================
    follow_in_app = Column(Boolean, default=True, nullable=False)
    follow_email = Column(Boolean, default=True, nullable=False)
    follow_push = Column(Boolean, default=True, nullable=False)
    
    like_in_app = Column(Boolean, default=True, nullable=False)
    like_email = Column(Boolean, default=False, nullable=False)
    like_push = Column(Boolean, default=True, nullable=False)
    
    comment_in_app = Column(Boolean, default=True, nullable=False)
    comment_email = Column(Boolean, default=True, nullable=False)
    comment_push = Column(Boolean, default=True, nullable=False)
    
    mention_in_app = Column(Boolean, default=True, nullable=False)
    mention_email = Column(Boolean, default=True, nullable=False)
    mention_push = Column(Boolean, default=True, nullable=False)
    
    repost_in_app = Column(Boolean, default=True, nullable=False)
    repost_email = Column(Boolean, default=False, nullable=False)
    repost_push = Column(Boolean, default=True, nullable=False)
    
    # ==================== Video Notifications ====================
    video_upload_in_app = Column(Boolean, default=True, nullable=False)
    video_upload_email = Column(Boolean, default=True, nullable=False)
    video_upload_push = Column(Boolean, default=True, nullable=False)
    
    video_processed_in_app = Column(Boolean, default=True, nullable=False)
    video_processed_email = Column(Boolean, default=False, nullable=False)
    video_processed_push = Column(Boolean, default=True, nullable=False)
    
    # ==================== Live Streaming Notifications ====================
    livestream_start_in_app = Column(Boolean, default=True, nullable=False)
    livestream_start_email = Column(Boolean, default=True, nullable=False)
    livestream_start_push = Column(Boolean, default=True, nullable=False)
    
    stream_donation_in_app = Column(Boolean, default=True, nullable=False)
    stream_donation_email = Column(Boolean, default=True, nullable=False)
    stream_donation_push = Column(Boolean, default=True, nullable=False)
    
    # ==================== Payment Notifications ====================
    payment_in_app = Column(Boolean, default=True, nullable=False)
    payment_email = Column(Boolean, default=True, nullable=False)
    payment_push = Column(Boolean, default=True, nullable=False)
    payment_sms = Column(Boolean, default=False, nullable=False)
    
    payout_in_app = Column(Boolean, default=True, nullable=False)
    payout_email = Column(Boolean, default=True, nullable=False)
    payout_push = Column(Boolean, default=True, nullable=False)
    payout_sms = Column(Boolean, default=False, nullable=False)
    
    subscription_in_app = Column(Boolean, default=True, nullable=False)
    subscription_email = Column(Boolean, default=True, nullable=False)
    subscription_push = Column(Boolean, default=True, nullable=False)
    
    # ==================== Moderation Notifications ====================
    moderation_in_app = Column(Boolean, default=True, nullable=False)
    moderation_email = Column(Boolean, default=True, nullable=False)
    moderation_push = Column(Boolean, default=True, nullable=False)
    
    # ==================== System Notifications ====================
    system_in_app = Column(Boolean, default=True, nullable=False)
    system_email = Column(Boolean, default=True, nullable=False)
    system_push = Column(Boolean, default=True, nullable=False)
    
    security_in_app = Column(Boolean, default=True, nullable=False)
    security_email = Column(Boolean, default=True, nullable=False)
    security_push = Column(Boolean, default=True, nullable=False)
    security_sms = Column(Boolean, default=True, nullable=False)
    
    # ==================== Global Settings ====================
    do_not_disturb = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Pause all notifications"
    )
    
    do_not_disturb_start = Column(
        String(5),
        nullable=True,
        doc="DND start time (HH:MM format)"
    )
    
    do_not_disturb_end = Column(
        String(5),
        nullable=True,
        doc="DND end time (HH:MM format)"
    )
    
    # ==================== Digest Settings ====================
    email_digest_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable email digest"
    )
    
    email_digest_frequency = Column(
        String(20),
        default="daily",
        nullable=False,
        doc="Digest frequency (daily, weekly)"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        backref="notification_settings",
        foreign_keys=[user_id]
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_notification_settings_user', 'user_id'),
    )
    
    def __repr__(self) -> str:
        return f"<NotificationSettings(user_id={self.user_id})>"
    
    def is_channel_enabled(
        self,
        notification_type: NotificationType,
        channel: NotificationChannel
    ) -> bool:
        """
        Check if a specific notification type/channel combination is enabled.
        
        Args:
            notification_type: Type of notification
            channel: Delivery channel
            
        Returns:
            True if enabled, False otherwise
        """
        # Map notification types to setting prefixes
        type_map = {
            NotificationType.FOLLOW: "follow",
            NotificationType.LIKE: "like",
            NotificationType.COMMENT: "comment",
            NotificationType.MENTION: "mention",
            NotificationType.REPOST: "repost",
            NotificationType.VIDEO_UPLOADED: "video_upload",
            NotificationType.VIDEO_PROCESSED: "video_processed",
            NotificationType.LIVE_STREAM_STARTED: "livestream_start",
            NotificationType.STREAM_DONATION: "stream_donation",
            NotificationType.PAYMENT_RECEIVED: "payment",
            NotificationType.PAYOUT_PROCESSED: "payout",
            NotificationType.SUBSCRIPTION_STARTED: "subscription",
            NotificationType.CONTENT_FLAGGED: "moderation",
            NotificationType.SYSTEM_ANNOUNCEMENT: "system",
            NotificationType.SECURITY_ALERT: "security",
        }
        
        # Get setting prefix
        prefix = type_map.get(notification_type)
        if not prefix:
            return True  # Default to enabled if type not mapped
        
        # Build attribute name
        channel_suffix = channel.value
        attr_name = f"{prefix}_{channel_suffix}"
        
        # Get setting value
        return getattr(self, attr_name, True)


class PushToken(CommonBase):
    """
    Push notification token model.
    
    Stores FCM/APNS tokens for push notifications.
    """
    
    __tablename__ = "push_tokens"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User ID"
    )
    
    # ==================== Token ====================
    token = Column(
        String(500),
        nullable=False,
        unique=True,
        index=True,
        doc="Push notification token"
    )
    
    # ==================== Platform ====================
    platform = Column(
        SQLEnum(PushPlatform),
        nullable=False,
        index=True,
        doc="Push platform"
    )
    
    # ==================== Device Information ====================
    device_id = Column(
        String(255),
        nullable=True,
        index=True,
        doc="Device identifier"
    )
    
    device_name = Column(
        String(100),
        nullable=True,
        doc="Device name"
    )
    
    device_model = Column(
        String(100),
        nullable=True,
        doc="Device model"
    )
    
    os_version = Column(
        String(50),
        nullable=True,
        doc="OS version"
    )
    
    app_version = Column(
        String(50),
        nullable=True,
        doc="App version"
    )
    
    # ==================== Status ====================
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether token is active"
    )
    
    # ==================== Metadata ====================
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last time token was used"
    )
    
    failed_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of consecutive failures"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        backref="push_tokens",
        foreign_keys=[user_id]
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_push_token_user', 'user_id', 'is_active'),
        Index('idx_push_token_platform', 'platform', 'is_active'),
        Index('idx_push_token_device', 'device_id', 'user_id'),
    )
    
    def __repr__(self) -> str:
        return f"<PushToken(id={self.id}, user_id={self.user_id}, platform={self.platform})>"
    
    def mark_as_failed(self) -> None:
        """Mark token as failed and increment failure count."""
        self.failed_count += 1
        if self.failed_count >= 3:
            self.is_active = False
    
    def mark_as_used(self) -> None:
        """Mark token as successfully used."""
        from datetime import datetime, timezone as tz
        self.last_used_at = datetime.now(tz.utc)
        self.failed_count = 0
        self.is_active = True


# Export models
__all__ = [
    'Notification',
    'NotificationSettings',
    'PushToken',
    'NotificationType',
    'NotificationChannel',
    'NotificationStatus',
    'PushPlatform',
]
