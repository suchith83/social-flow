"""
Follow model and related schemas.

This module defines the Follow model and related Pydantic schemas
for follow management in the Social Flow backend.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Follow(Base):
    """Follow model for storing user follow relationships."""
    
    __tablename__ = "follows"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Foreign keys
    follower_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    following_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationships
    follower = relationship("User", foreign_keys=[follower_id], back_populates="following")
    following = relationship("User", foreign_keys=[following_id], back_populates="followers")
    
    def __repr__(self) -> str:
        return f"<Follow(id={self.id}, follower_id={self.follower_id}, following_id={self.following_id})>"
    
    def to_dict(self) -> dict:
        """Convert follow to dictionary."""
        return {
            "id": str(self.id),
            "follower_id": str(self.follower_id),
            "following_id": str(self.following_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
