"""
View count model and related schemas.

This module defines the ViewCount model and related Pydantic schemas
for view count tracking in the Social Flow backend.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ViewCount(Base):
    """ViewCount model for tracking view counts."""
    
    __tablename__ = "view_counts"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # View count information
    count = Column(Integer, default=0, nullable=False)
    unique_views = Column(Integer, default=0, nullable=False)
    
    # Time period information
    period = Column(String(20), nullable=False)  # daily, weekly, monthly, total
    date = Column(DateTime, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    video_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # For user-specific views
    
    # Relationships
    video = relationship("Video", back_populates="view_counts")
    
    def __repr__(self) -> str:
        return f"<ViewCount(id={self.id}, video_id={self.video_id}, count={self.count}, period={self.period})>"
    
    def to_dict(self) -> dict:
        """Convert view count to dictionary."""
        return {
            "id": str(self.id),
            "count": self.count,
            "unique_views": self.unique_views,
            "period": self.period,
            "date": self.date.isoformat() if self.date else None,
            "video_id": str(self.video_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
