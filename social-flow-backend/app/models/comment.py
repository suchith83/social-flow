"""
Comment model and related schemas.

This module defines the Comment model and related Pydantic schemas
for comment management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Comment(Base):
    """Comment model for storing comments on posts and videos."""
    
    __tablename__ = "comments"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Content
    content = Column(Text, nullable=False)
    hashtags = Column(Text, nullable=True)  # JSON string of hashtags
    mentions = Column(Text, nullable=True)  # JSON string of mentions
    
    # Engagement metrics
    likes_count = Column(Integer, default=0, nullable=False)
    replies_count = Column(Integer, default=0, nullable=False)
    
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
    
    # Reply information
    is_reply = Column(Boolean, default=False, nullable=False)
    parent_comment_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    owner_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    post_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    video_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Relationships
    owner = relationship("User", back_populates="comments")
    post = relationship("Post", back_populates="comments")
    video = relationship("Video", back_populates="comments")
    likes = relationship("Like", back_populates="comment", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Comment(id={self.id}, content={self.content[:50]}..., owner_id={self.owner_id})>"
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.likes_count == 0:
            return 0.0
        
        return (self.likes_count / (self.likes_count + self.replies_count)) * 100
    
    def to_dict(self) -> dict:
        """Convert comment to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "likes_count": self.likes_count,
            "replies_count": self.replies_count,
            "is_approved": self.is_approved,
            "is_flagged": self.is_flagged,
            "is_rejected": self.is_rejected,
            "is_reply": self.is_reply,
            "parent_comment_id": str(self.parent_comment_id) if self.parent_comment_id else None,
            "engagement_rate": self.engagement_rate,
            "owner_id": str(self.owner_id),
            "post_id": str(self.post_id) if self.post_id else None,
            "video_id": str(self.video_id) if self.video_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
