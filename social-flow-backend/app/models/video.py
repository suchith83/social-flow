"""
Video model and related schemas.

This module defines the Video model and related Pydantic schemas
for video management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Enum as SQLEnum, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class VideoStatus(str, Enum):
    """Video processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class VideoVisibility(str, Enum):
    """Video visibility level."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


class Video(Base):
    """Video model for storing video information."""
    
    __tablename__ = "videos"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)  # JSON string of tags
    
    # File information
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)  # Duration in seconds
    resolution = Column(String(20), nullable=True)  # e.g., "1920x1080"
    bitrate = Column(Integer, nullable=True)  # Bitrate in kbps
    codec = Column(String(50), nullable=True)  # e.g., "h264"
    
    # Storage information
    s3_key = Column(String(500), nullable=False)
    s3_bucket = Column(String(100), nullable=False)
    thumbnail_url = Column(String(500), nullable=True)
    preview_url = Column(String(500), nullable=True)
    
    # Streaming information
    hls_url = Column(String(500), nullable=True)
    dash_url = Column(String(500), nullable=True)
    streaming_url = Column(String(500), nullable=True)
    
    # Status and visibility
    status = Column(SQLEnum(VideoStatus), default=VideoStatus.UPLOADING, nullable=False)
    visibility = Column(SQLEnum(VideoVisibility), default=VideoVisibility.PUBLIC, nullable=False)
    
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
    
    # Engagement metrics
    views_count = Column(Integer, default=0, nullable=False)
    likes_count = Column(Integer, default=0, nullable=False)
    dislikes_count = Column(Integer, default=0, nullable=False)
    comments_count = Column(Integer, default=0, nullable=False)
    shares_count = Column(Integer, default=0, nullable=False)
    
    # Watch time metrics
    total_watch_time = Column(Float, default=0.0, nullable=False)  # Total watch time in seconds
    average_watch_time = Column(Float, default=0.0, nullable=False)  # Average watch time in seconds
    retention_rate = Column(Float, default=0.0, nullable=False)  # Retention rate as percentage
    
    # Monetization
    is_monetized = Column(Boolean, default=False, nullable=False)
    ad_revenue = Column(Float, default=0.0, nullable=False)
    
    # Processing information
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    owner_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationships
    owner = relationship("User", back_populates="videos")
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="video", cascade="all, delete-orphan")
    view_counts = relationship("ViewCount", back_populates="video", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Video(id={self.id}, title={self.title}, owner_id={self.owner_id})>"
    
    @property
    def is_processing(self) -> bool:
        """Check if video is currently being processed."""
        return self.status in [VideoStatus.UPLOADING, VideoStatus.PROCESSING]
    
    @property
    def is_ready(self) -> bool:
        """Check if video is ready for viewing."""
        return self.status == VideoStatus.PROCESSED and self.is_approved
    
    @property
    def is_public(self) -> bool:
        """Check if video is publicly visible."""
        return self.visibility == VideoVisibility.PUBLIC and self.is_ready
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.views_count == 0:
            return 0.0
        
        total_engagement = self.likes_count + self.comments_count + self.shares_count
        return (total_engagement / self.views_count) * 100
    
    @property
    def like_ratio(self) -> float:
        """Calculate like ratio."""
        total_reactions = self.likes_count + self.dislikes_count
        if total_reactions == 0:
            return 0.0
        
        return (self.likes_count / total_reactions) * 100
    
    def to_dict(self) -> dict:
        """Convert video to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "filename": self.filename,
            "file_size": self.file_size,
            "duration": self.duration,
            "resolution": self.resolution,
            "bitrate": self.bitrate,
            "codec": self.codec,
            "s3_key": self.s3_key,
            "s3_bucket": self.s3_bucket,
            "thumbnail_url": self.thumbnail_url,
            "preview_url": self.preview_url,
            "hls_url": self.hls_url,
            "dash_url": self.dash_url,
            "streaming_url": self.streaming_url,
            "status": self.status.value,
            "visibility": self.visibility.value,
            "is_approved": self.is_approved,
            "is_flagged": self.is_flagged,
            "is_rejected": self.is_rejected,
            "views_count": self.views_count,
            "likes_count": self.likes_count,
            "dislikes_count": self.dislikes_count,
            "comments_count": self.comments_count,
            "shares_count": self.shares_count,
            "total_watch_time": self.total_watch_time,
            "average_watch_time": self.average_watch_time,
            "retention_rate": self.retention_rate,
            "is_monetized": self.is_monetized,
            "ad_revenue": self.ad_revenue,
            "engagement_rate": self.engagement_rate,
            "like_ratio": self.like_ratio,
            "owner_id": str(self.owner_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
        }
