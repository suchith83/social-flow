"""
Post model and related schemas.

This module defines the Post model and related Pydantic schemas
for post management in the Social Flow backend.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Post(Base):
    """Post model for storing social media posts."""
    
    __tablename__ = "posts"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Content
    content = Column(Text, nullable=False)
    media_url = Column(String(500), nullable=True)
    media_type = Column(String(50), nullable=True)  # image, video, gif
    hashtags = Column(Text, nullable=True)  # JSON string of hashtags
    mentions = Column(Text, nullable=True)  # JSON string of mentions
    
    # Engagement metrics
    likes_count = Column(Integer, default=0, nullable=False)
    reposts_count = Column(Integer, default=0, nullable=False)
    comments_count = Column(Integer, default=0, nullable=False)
    shares_count = Column(Integer, default=0, nullable=False)
    views_count = Column(Integer, default=0, nullable=False)
    
    # Moderation
    is_approved = Column(Boolean, default=False, nullable=False)
    is_flagged = Column(Boolean, default=False, nullable=False)
    is_rejected = Column(Boolean, default=False, nullable=False)
    
    # Moderation details
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(UUID(as_uuid=True), nullable=True)
    flagged_at = Column(DateTime, nullable=True)
    flagged_by = Column(UUID(as_uuid=True), nullable=True)
    flag_reason = Column(Text, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    rejected_by = Column(UUID(as_uuid=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Repost information
    is_repost = Column(Boolean, default=False, nullable=False)
    original_post_id = Column(UUID(as_uuid=True), nullable=True)
    repost_reason = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Relationships
    owner = relationship("User", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="post", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Post(id={self.id}, content={self.content[:50]}..., owner_id={self.owner_id})>"
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.views_count == 0:
            return 0.0
        
        total_engagement = self.likes_count + self.reposts_count + self.comments_count + self.shares_count
        return (total_engagement / self.views_count) * 100
    
    def to_dict(self) -> dict:
        """Convert post to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content,
            "media_url": self.media_url,
            "media_type": self.media_type,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "likes_count": self.likes_count,
            "reposts_count": self.reposts_count,
            "comments_count": self.comments_count,
            "shares_count": self.shares_count,
            "views_count": self.views_count,
            "is_approved": self.is_approved,
            "is_flagged": self.is_flagged,
            "is_rejected": self.is_rejected,
            "is_repost": self.is_repost,
            "original_post_id": str(self.original_post_id) if self.original_post_id else None,
            "repost_reason": self.repost_reason,
            "engagement_rate": self.engagement_rate,
            "owner_id": str(self.owner_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
