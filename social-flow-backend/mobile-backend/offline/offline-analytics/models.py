# Pydantic/ORM models for events and aggregates
"""
SQLAlchemy models (DB) and Pydantic schemas (API) for offline analytics.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, func, Index, BigInteger
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import enum

Base = declarative_base()


class IngestStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class RawEvent(Base):
    """
    Stores raw event blobs uploaded by clients (usually batched when reconnecting).
    Each record represents one event or an item from a batch depending on ingestion.
    """
    __tablename__ = "raw_events"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(128), nullable=True, index=True)
    user_id = Column(String(128), nullable=True, index=True)
    event_type = Column(String(128), nullable=False, index=True)
    event_payload = Column(JSON, nullable=True)
    occurred_at = Column(DateTime, nullable=False)
    ingested_at = Column(DateTime, server_default=func.now())
    status = Column(String(32), default=IngestStatus.PENDING.value, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    size_bytes = Column(BigInteger, nullable=False, default=0)


class AggregatedMetric(Base):
    """
    Simple time-bucketed aggregation store (example).
    """
    __tablename__ = "aggregated_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(128), nullable=False, index=True)
    bucket_start = Column(DateTime, nullable=False, index=True)
    bucket_end = Column(DateTime, nullable=False)
    value = Column(Integer, nullable=False, default=0)
    metadata = Column(JSON, nullable=True)


Index("idx_metric_bucket", AggregatedMetric.metric_name, AggregatedMetric.bucket_start)


# -------------------- Pydantic schemas --------------------

class EventPayload(BaseModel):
    # flexible payload; part of schema for validation
    data: Dict[str, Any] = Field(default_factory=dict)


class RawEventCreate(BaseModel):
    device_id: Optional[str] = Field(None, description="Client device id")
    user_id: Optional[str] = Field(None, description="User id (if available)")
    event_type: str = Field(..., description="Event type/name")
    event_payload: dict = Field(default_factory=dict)
    occurred_at: str = Field(..., description="ISO datetime when event occurred")


class RawEventRead(BaseModel):
    id: int
    device_id: Optional[str]
    user_id: Optional[str]
    event_type: str
    event_payload: dict
    occurred_at: str
    ingested_at: str
    status: str
    processed_at: Optional[str]
    size_bytes: int

    class Config:
        orm_mode = True


class BatchIngestRequest(BaseModel):
    events: List[RawEventCreate] = Field(..., description="List of events sent by client")
