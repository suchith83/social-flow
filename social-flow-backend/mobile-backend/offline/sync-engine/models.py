# Pydantic or ORM models for sync jobs
"""
SQLAlchemy models and Pydantic schemas for the sync engine.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column, Integer, String, DateTime, JSON, Boolean, func, BigInteger, Index
)
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import enum
from datetime import datetime

Base = declarative_base()


class ChangeType(str, enum.Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class SyncItem(Base):
    """
    Canonical server-side item representation.
    - key: unique id known by app (e.g., "note:123" or UUID)
    - data: JSON payload (application-specific)
    - server_version: monotonic server version (Lamport-like)
    - client_version: optional last-known client version for optimistic checks
    - deleted: tombstone flag
    """
    __tablename__ = "sync_items"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(512), unique=True, nullable=False, index=True)
    data = Column(JSON, nullable=True)
    server_version = Column(BigInteger, nullable=False, default=0)
    client_version = Column(BigInteger, nullable=False, default=0)
    deleted = Column(Boolean, default=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())


class ChangeLog(Base):
    """
    Ordered change log of operations that clients can consume incrementally.
    Each change has a global monotonically-increasing sequence number.
    """
    __tablename__ = "change_log"

    seq = Column(BigInteger, primary_key=True, index=True)  # global sequence
    key = Column(String(512), nullable=False, index=True)
    change_type = Column(String(32), nullable=False)
    payload = Column(JSON, nullable=True)  # new state or delta
    server_version = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), index=True)


Index("idx_changelog_created", ChangeLog.created_at)


# ---------------- Pydantic Schemas ----------------

class ItemPayload(BaseModel):
    key: str = Field(..., description="Unique item key")
    data: Dict[str, Any] = Field(default_factory=dict, description="Item data")
    client_version: Optional[int] = Field(None, description="Client's last-known version")

class PushOperation(BaseModel):
    """
    One operation from a client in a push batch.
    type: create|update|delete
    """
    type: ChangeType
    item: ItemPayload

class PushBatchRequest(BaseModel):
    client_id: str = Field(..., description="Unique client ID")
    last_sync_seq: Optional[int] = Field(0, description="Last global sequence the client has applied")
    ops: List[PushOperation] = Field(..., description="Ordered ops in the batch")

class PullRequest(BaseModel):
    client_id: str
    since_seq: int = Field(0, description="Sequence number after which to return changes")
    page_size: Optional[int] = Field(100)


class ChangeRecord(BaseModel):
    seq: int
    key: str
    change_type: ChangeType
    payload: Optional[dict]
    server_version: int
    created_at: datetime

class PullResponse(BaseModel):
    changes: List[ChangeRecord]
    last_seq: int
