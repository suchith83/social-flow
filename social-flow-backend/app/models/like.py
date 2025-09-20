"""
Like model and related schemas.

This module defines the Like model and related Pydantic schemas
for like management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Like(Base):
    """Like model for storing likes on posts, videos, and comments."""
    
    __tablename__ = "likes"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Like type and status
    like_type = Column(String(20), nullable=False)  # post, video, comment
    is_like = Column(Boolean, default=True, nullable=False)  # True for like, False for dislike
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    post_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    video_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    comment_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Relationships
    user = relationship("User", back_populates="likes")
    post = relationship("Post", back_populates="likes")
    video = relationship("Video", back_populates="likes")
    comment = relationship("Comment", back_populates="likes")
    
    def __repr__(self) -> str:
        return f"<Like(id={self.id}, user_id={self.user_id}, type={self.like_type}, is_like={self.is_like})>"
    
    def to_dict(self) -> dict:
        """Convert like to dictionary."""
        return {
            "id": str(self.id),
            "like_type": self.like_type,
            "is_like": self.is_like,
            "user_id": str(self.user_id),
            "post_id": str(self.post_id) if self.post_id else None,
            "video_id": str(self.video_id) if self.video_id else None,
            "comment_id": str(self.comment_id) if self.comment_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
