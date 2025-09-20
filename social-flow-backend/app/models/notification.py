"""
Notification model and related schemas.

This module defines the Notification model and related Pydantic schemas
for notification management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class NotificationType(str, Enum):
    """Notification type."""
    LIKE = "like"
    COMMENT = "comment"
    FOLLOW = "follow"
    MENTION = "mention"
    SHARE = "share"
    VIDEO_UPLOAD = "video_upload"
    POST_CREATE = "post_create"
    SUBSCRIPTION = "subscription"
    PAYMENT = "payment"
    SYSTEM = "system"
    AD = "ad"


class NotificationStatus(str, Enum):
    """Notification status."""
    UNREAD = "unread"
    READ = "read"
    ARCHIVED = "archived"


class Notification(Base):
    """Notification model for storing notification information."""
    
    __tablename__ = "notifications"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Notification content
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)
    status = Column(String(20), default=NotificationStatus.UNREAD, nullable=False)
    
    # Action information
    action_url = Column(String(500), nullable=True)
    action_text = Column(String(100), nullable=True)
    
    # Entity information
    entity_type = Column(String(50), nullable=True)  # post, video, comment, etc.
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Sender information
    sender_id = Column(UUID(as_uuid=True), nullable=True)
    sender_name = Column(String(255), nullable=True)
    sender_avatar_url = Column(String(500), nullable=True)
    
    # Metadata
    metadata = Column(Text, nullable=True)  # JSON string of additional data
    
    # Delivery information
    is_push_sent = Column(Boolean, default=False, nullable=False)
    is_email_sent = Column(Boolean, default=False, nullable=False)
    push_sent_at = Column(DateTime, nullable=True)
    email_sent_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    read_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    
    def __repr__(self) -> str:
        return f"<Notification(id={self.id}, title={self.title}, user_id={self.user_id})>"
    
    @property
    def is_read(self) -> bool:
        """Check if notification is read."""
        return self.status == NotificationStatus.READ
    
    @property
    def is_archived(self) -> bool:
        """Check if notification is archived."""
        return self.status == NotificationStatus.ARCHIVED
    
    @property
    def is_unread(self) -> bool:
        """Check if notification is unread."""
        return self.status == NotificationStatus.UNREAD
    
    def to_dict(self) -> dict:
        """Convert notification to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "message": self.message,
            "notification_type": self.notification_type,
            "status": self.status,
            "action_url": self.action_url,
            "action_text": self.action_text,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "sender_id": str(self.sender_id) if self.sender_id else None,
            "sender_name": self.sender_name,
            "sender_avatar_url": self.sender_avatar_url,
            "metadata": self.metadata,
            "is_push_sent": self.is_push_sent,
            "is_email_sent": self.is_email_sent,
            "push_sent_at": self.push_sent_at.isoformat() if self.push_sent_at else None,
            "email_sent_at": self.email_sent_at.isoformat() if self.email_sent_at else None,
            "is_read": self.is_read,
            "is_archived": self.is_archived,
            "is_unread": self.is_unread,
            "user_id": str(self.user_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
        }
