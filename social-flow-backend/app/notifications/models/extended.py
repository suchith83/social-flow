"""
Extended Notification Models

Enhanced database models for comprehensive notification system.
Extends existing notification.py with preferences, templates, and tracking.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from app.core.database import Base


class NotificationPreference(Base):
    """
    User notification preferences.
    
    Controls which notification types are enabled for each channel.
    """
    
    __tablename__ = "notification_preferences"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # Global settings
    email_enabled = Column(Boolean, default=True, nullable=False)
    push_enabled = Column(Boolean, default=True, nullable=False)
    sms_enabled = Column(Boolean, default=False, nullable=False)
    
    # Engagement notifications
    new_follower_enabled = Column(Boolean, default=True, nullable=False)
    new_like_enabled = Column(Boolean, default=True, nullable=False)
    new_comment_enabled = Column(Boolean, default=True, nullable=False)
    mention_enabled = Column(Boolean, default=True, nullable=False)
    
    # Content notifications
    video_processing_enabled = Column(Boolean, default=True, nullable=False)
    live_stream_enabled = Column(Boolean, default=True, nullable=False)
    
    # Moderation notifications
    moderation_enabled = Column(Boolean, default=True, nullable=False)
    copyright_enabled = Column(Boolean, default=True, nullable=False)
    
    # Payment notifications
    payment_enabled = Column(Boolean, default=True, nullable=False)
    payout_enabled = Column(Boolean, default=True, nullable=False)
    donation_enabled = Column(Boolean, default=True, nullable=False)
    
    # System notifications
    system_enabled = Column(Boolean, default=True, nullable=False)
    security_enabled = Column(Boolean, default=True, nullable=False)
    
    # Digest settings
    daily_digest_enabled = Column(Boolean, default=False, nullable=False)
    weekly_digest_enabled = Column(Boolean, default=True, nullable=False)
    digest_time = Column(String(5), default="09:00", nullable=False)  # HH:MM format
    
    # Quiet hours
    quiet_hours_enabled = Column(Boolean, default=False, nullable=False)
    quiet_hours_start = Column(String(5), nullable=True)  # HH:MM
    quiet_hours_end = Column(String(5), nullable=True)    # HH:MM
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    
    # Relationships
    user = relationship("User", back_populates="notification_preferences")
    
    def __repr__(self):
        return f"<NotificationPreference(user_id={self.user_id})>"


class NotificationTemplate(Base):
    """
    Reusable notification templates.
    
    Defines template structure for each notification type.
    """
    
    __tablename__ = "notification_templates"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Template details
    type = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Content templates (support variable substitution)
    title_template = Column(String(200), nullable=False)
    message_template = Column(Text, nullable=False)
    
    # Email templates
    email_subject_template = Column(String(200), nullable=True)
    email_body_template = Column(Text, nullable=True)
    email_html_template = Column(Text, nullable=True)
    
    # Push notification templates
    push_title_template = Column(String(200), nullable=True)
    push_body_template = Column(Text, nullable=True)
    
    # Default values
    default_icon = Column(String(100), nullable=True)
    default_action_label = Column(String(100), nullable=True)
    default_priority = Column(String(20), default="normal", nullable=False)
    default_channels = Column(ARRAY(String), default=["in_app"], nullable=False)
    
    # Expiry
    default_expires_hours = Column(Integer, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    
    def __repr__(self):
        return f"<NotificationTemplate(type={self.type}, name={self.name})>"


class EmailLog(Base):
    """
    Email delivery tracking.
    
    Logs all emails sent for delivery tracking and troubleshooting.
    """
    
    __tablename__ = "email_logs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Notification reference
    notification_id = Column(UUID(as_uuid=True), ForeignKey("notifications.id", ondelete="CASCADE"), nullable=True)
    
    # Recipient
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    to_email = Column(String(255), nullable=False)
    
    # Email details
    subject = Column(String(200), nullable=False)
    body_text = Column(Text, nullable=True)
    body_html = Column(Text, nullable=True)
    
    # Provider details
    provider = Column(String(50), default="sendgrid", nullable=False)
    provider_message_id = Column(String(200), nullable=True)
    
    # Status
    status = Column(String(20), default="pending", nullable=False, index=True)
    
    # Delivery tracking
    sent_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    clicked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Error tracking
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    bounce_type = Column(String(50), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    notification = relationship("Notification")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_email_logs_user_created', 'user_id', 'created_at'),
        Index('idx_email_logs_status', 'status'),
        Index('idx_email_logs_provider_id', 'provider_message_id'),
    )
    
    def __repr__(self):
        return f"<EmailLog(id={self.id}, to={self.to_email}, status={self.status})>"


class PushNotificationToken(Base):
    """
    Mobile device push notification tokens.
    
    Stores FCM/APNS tokens for push notifications.
    """
    
    __tablename__ = "push_notification_tokens"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Device details
    device_id = Column(String(200), nullable=False)
    device_type = Column(String(20), nullable=False)  # ios, android
    device_name = Column(String(200), nullable=True)
    
    # Token
    token = Column(String(500), nullable=False, unique=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Usage tracking
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    
    # Relationships
    user = relationship("User", back_populates="push_tokens")
    
    # Indexes
    __table_args__ = (
        Index('idx_push_tokens_user', 'user_id', 'is_active'),
        Index('idx_push_tokens_device', 'device_id'),
    )
    
    def __repr__(self):
        return f"<PushNotificationToken(user_id={self.user_id}, device_type={self.device_type})>"
