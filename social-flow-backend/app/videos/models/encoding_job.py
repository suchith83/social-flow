"""
Encoding Job Model for video transcoding jobs.
"""

import enum
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from sqlalchemy import JSON, Column, Integer, String, DateTime, Enum as SQLEnum, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class EncodingStatus(str, enum.Enum):
    """Encoding job status."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class EncodingQuality(str, enum.Enum):
    """Video encoding quality levels."""
    MOBILE_240P = "240p"
    SD_360P = "360p"
    SD_480P = "480p"
    HD_720P = "720p"
    FULL_HD_1080P = "1080p"
    UHD_4K = "4k"


class EncodingJob(Base):
    """
    Encoding job model for tracking video transcoding jobs.
    
    Tracks the status of AWS MediaConvert jobs and associated metadata.
    Supports both cloud (MediaConvert) and local (FFmpeg) encoding.
    """
    __tablename__ = "encoding_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Input/output paths
    input_path = Column(String(500), nullable=False)  # S3 key or local path
    input_s3_key = Column(String(500), nullable=True)  # Legacy field
    output_s3_prefix = Column(String(500), nullable=True)  # Legacy field
    output_format = Column(String(50), default="hls")  # hls, dash
    output_paths = Column(JSON, nullable=True)  # Dict of quality -> S3 path
    
    # Manifest URLs for adaptive streaming
    hls_manifest_url = Column(String(500), nullable=True)
    dash_manifest_url = Column(String(500), nullable=True)
    
    # AWS MediaConvert job ID (if cloud encoding)
    mediaconvert_job_id = Column(String(200), nullable=True, index=True)
    
    # Status and progress
    status = Column(SQLEnum(EncodingStatus), default=EncodingStatus.PENDING, nullable=False, index=True)
    progress = Column(Integer, default=0)  # 0-100
    
    # Quality levels (JSON array)
    qualities = Column(Text, nullable=True)  # JSON array of quality levels
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="encoding_jobs")
    
    def __repr__(self) -> str:
        return f"<EncodingJob(id={self.id}, video_id={self.video_id}, status={self.status})>"

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate encoding duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_completed(self) -> bool:
        """Check if encoding is completed."""
        return self.status == EncodingStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if encoding has failed."""
        return self.status == EncodingStatus.FAILED

    @property
    def is_processing(self) -> bool:
        """Check if encoding is currently processing."""
        return self.status in (EncodingStatus.PROCESSING, EncodingStatus.QUEUED)
