# Schemas for requests/responses & DB models
"""
Database models & Pydantic schemas for background sync API.
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Enum, func
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field
from typing import Optional, List
import enum

Base = declarative_base()


class SyncStatus(str, enum.Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class SyncJob(Base):
    __tablename__ = "sync_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String(255), nullable=False)
    status = Column(Enum(SyncStatus), default=SyncStatus.PENDING, nullable=False)
    payload = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


# -------------------- Pydantic Schemas --------------------

class SyncJobCreate(BaseModel):
    job_name: str = Field(..., description="Unique name of the sync job")
    payload: dict = Field(default_factory=dict, description="Payload for background sync")


class SyncJobRead(BaseModel):
    id: int
    job_name: str
    status: SyncStatus
    payload: Optional[dict]
    result: Optional[dict]
    created_at: str
    updated_at: Optional[str]

    class Config:
        orm_mode = True


class SyncJobListResponse(BaseModel):
    jobs: List[SyncJobRead]
