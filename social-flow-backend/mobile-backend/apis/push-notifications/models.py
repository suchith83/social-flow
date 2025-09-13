# Pydantic schemas for notifications and DB models
"""
SQLAlchemy models + Pydantic schemas for push notifications.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    Boolean,
    func,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()


class DevicePlatform(str, enum.Enum):
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"


class NotificationStatus(str, enum.Enum):
    PENDING = "PENDING"
    SENDING = "SENDING"
    SENT = "SENT"
    FAILED = "FAILED"


class Device(Base):
    __tablename__ = "push_devices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), nullable=True, index=True)  # optional mapping to user
    token = Column(String(512), nullable=False, unique=True, index=True)
    platform = Column(Enum(DevicePlatform), nullable=False)
    app_version = Column(String(64), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    is_active = Column(Boolean, default=True)


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    body = Column(String(1024), nullable=True)
    payload = Column(JSON, nullable=True)  # app-specific data
    status = Column(Enum(NotificationStatus), default=NotificationStatus.PENDING)
    provider_response = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class NotificationAttempt(Base):
    __tablename__ = "notification_attempts"
    id = Column(Integer, primary_key=True, index=True)
    notification_id = Column(Integer, nullable=False, index=True)
    device_token = Column(String(512), nullable=False)
    status = Column(String(64), nullable=False)
    response = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


# ---------------- Pydantic schemas ----------------

class DeviceRegisterRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user id associated with the device")
    token: str = Field(..., description="Platform device token")
    platform: DevicePlatform
    app_version: Optional[str] = None


class DeviceRead(BaseModel):
    id: int
    user_id: Optional[str]
    token: str
    platform: DevicePlatform
    app_version: Optional[str]
    is_active: bool

    class Config:
        orm_mode = True


class SendNotificationRequest(BaseModel):
    title: str = Field(..., description="Notification title (short)")
    body: Optional[str] = Field(None, description="Notification body (longer)")
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary JSON payload")
    tokens: Optional[List[str]] = Field(None, description="List of device tokens (mutually exclusive with user_ids)")
    user_ids: Optional[List[str]] = Field(None, description="List of user IDs to target (server will map to tokens)")
    schedule_at: Optional[str] = Field(None, description="ISO timestamp to schedule the notification")


class NotificationRead(BaseModel):
    id: int
    title: str
    body: Optional[str]
    payload: Optional[Dict[str, Any]]
    status: NotificationStatus

    class Config:
        orm_mode = True
