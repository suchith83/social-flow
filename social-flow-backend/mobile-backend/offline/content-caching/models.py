# Pydantic or ORM models for cached content
"""
SQLAlchemy models + Pydantic schemas describing cached content metadata.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, BigInteger, JSON, func, Boolean, Index
from pydantic import BaseModel, Field
from typing import Optional
import enum

Base = declarative_base()


class CacheOrigin(str, enum.Enum):
    REMOTE = "remote"   # fetched from remote origin server
    PREFETCH = "prefetch"  # prefetched proactively
    MANUAL = "manual"   # cached by manual request


class CachedItem(Base):
    __tablename__ = "cached_items"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(512), unique=True, nullable=False, index=True)  # e.g., content:id or url
    size_bytes = Column(BigInteger, nullable=False, default=0)
    checksum = Column(String(128), nullable=True)  # e.g., sha256
    origin = Column(String(50), nullable=False, default=CacheOrigin.REMOTE.value)
    ttl = Column(Integer, nullable=False)
    accessed_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    is_stale = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_cached_items_accessed_at", "accessed_at"),
    )


# Pydantic schemas

class CachedItemCreate(BaseModel):
    key: str = Field(..., description="Unique cache key (e.g., content:<id> or URL)")
    size_bytes: int = Field(..., ge=0)
    checksum: Optional[str] = Field(None)
    origin: Optional[CacheOrigin] = Field(CacheOrigin.REMOTE)
    ttl: Optional[int] = Field(None)
    metadata: Optional[dict] = Field(default_factory=dict)


class CachedItemRead(BaseModel):
    id: int
    key: str
    size_bytes: int
    checksum: Optional[str]
    origin: str
    ttl: int
    accessed_at: str
    created_at: str
    is_stale: bool
    metadata: Optional[dict]

    class Config:
        orm_mode = True
