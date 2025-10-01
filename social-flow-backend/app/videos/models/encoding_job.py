"""
Encoding Job Model for video transcoding jobs.
"""

import enum
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class EncodingStatus(str, enum.Enum):
    """Encoding job status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class EncodingJob(Base):
    """
    Encoding job model for tracking video transcoding jobs.
    
    Tracks the status of AWS MediaConvert jobs and associated metadata.
    """
    __tablename__ = "encoding_jobs"
    
    id = Column(String, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    input_s3_key = Column(String, nullable=False)
    output_s3_prefix = Column(String, nullable=False)
    mediaconvert_job_id = Column(String, nullable=True, index=True)
    
    status = Column(SQLEnum(EncodingStatus), default=EncodingStatus.QUEUED, nullable=False, index=True)
    progress = Column(Integer, default=0)  # 0-100
    
    qualities = Column(Text, nullable=False)  # JSON array of quality levels
    
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    video = relationship("Video", back_populates="encoding_jobs")
    
    def __repr__(self):
        return f"<EncodingJob(id={self.id}, video_id={self.video_id}, status={self.status})>"
