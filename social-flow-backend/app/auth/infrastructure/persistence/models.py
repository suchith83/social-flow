"""
Database Models - Auth Infrastructure

SQLAlchemy models for persistence.
Pure database representation without business logic.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserModel(Base):
    """
    User database model.
    
    This is a pure persistence model - no business logic.
    Business logic lives in the domain UserEntity.
    """
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=False)
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    website = Column(String(500), nullable=True)
    location = Column(String(100), nullable=True)
    
    # Account status (stored as string to match AccountStatus enum)
    account_status = Column(String(30), nullable=False, default="pending_verification")
    privacy_level = Column(String(20), nullable=False, default="public")
    
    # Suspension details (denormalized for query performance)
    suspension_reason = Column(Text, nullable=True)
    suspended_at = Column(DateTime, nullable=True)
    suspension_ends_at = Column(DateTime, nullable=True)
    
    # Ban details (denormalized for query performance)
    ban_reason = Column(Text, nullable=True)
    banned_at = Column(DateTime, nullable=True)
    
    # Social metrics
    followers_count = Column(Integer, default=0, nullable=False)
    following_count = Column(Integer, default=0, nullable=False)
    posts_count = Column(Integer, default=0, nullable=False)
    videos_count = Column(Integer, default=0, nullable=False)
    total_views = Column(Integer, default=0, nullable=False)
    total_likes = Column(Integer, default=0, nullable=False)
    
    # Payment/Stripe Integration
    stripe_customer_id = Column(String(255), nullable=True, unique=True, index=True)
    
    # Preferences
    email_notifications = Column(Boolean, default=True, nullable=False)
    push_notifications = Column(Boolean, default=True, nullable=False)
    
    # Role (stored as string to match UserRole enum)
    role = Column(String(20), nullable=False, default="user")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    
    # Version for optimistic locking
    version = Column(Integer, default=1, nullable=False)
    
    # Relationships (commented out to avoid circular dependencies during initial migration)
    # These will be re-enabled once other bounded contexts are migrated
    # videos = relationship("Video", back_populates="owner", cascade="all, delete-orphan")
    # posts = relationship("Post", back_populates="owner", cascade="all, delete-orphan")
    # comments = relationship("Comment", back_populates="owner", cascade="all, delete-orphan")
    # likes = relationship("Like", back_populates="user", cascade="all, delete-orphan")
    # followers = relationship("Follow", foreign_keys="Follow.following_id", back_populates="following")
    # following = relationship("Follow", foreign_keys="Follow.follower_id", back_populates="follower")
    # payments = relationship("Payment", back_populates="user", cascade="all, delete-orphan")
    # subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    # notifications = relationship("Notification", back_populates="user", foreign_keys="[Notification.user_id]")
    # live_streams = relationship("LiveStream", back_populates="user", foreign_keys="[LiveStream.user_id]")
    
    def __repr__(self) -> str:
        return f"<UserModel(id={self.id}, username={self.username}, email={self.email})>"
