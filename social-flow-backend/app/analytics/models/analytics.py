"""
Analytics model and related schemas.

This module defines the Analytics model and related Pydantic schemas
for analytics tracking in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, Float, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base


class AnalyticsType(str, Enum):
    """Analytics event type."""
    PAGE_VIEW = "page_view"
    VIDEO_VIEW = "video_view"
    POST_VIEW = "post_view"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE = "performance"
    ERROR = "error"


class AnalyticsCategory(str, Enum):
    """Analytics event category."""
    USER = "user"
    CONTENT = "content"
    TECHNICAL = "technical"
    BUSINESS = "business"


class Analytics(Base):
    """Analytics model for storing analytics events."""
    
    __tablename__ = "analytics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event information
    event_type = Column(String(50), nullable=False)
    category = Column(String(50), nullable=False)
    event = Column(String(100), nullable=False)
    
    # Entity information
    entity_type = Column(String(50), nullable=True)  # user, video, post, etc.
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    
    # User information
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    session_id = Column(String(255), nullable=True)
    
    # Event data
    properties = Column(Text, nullable=True)  # JSON string of event properties
    context = Column(Text, nullable=True)  # JSON string of event context
    
    # Device and browser information
    user_agent = Column(Text, nullable=True)
    device = Column(String(100), nullable=True)
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)
    
    # Location information
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    country = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    # Referrer information
    referrer = Column(String(500), nullable=True)
    referrer_domain = Column(String(255), nullable=True)
    
    # Performance metrics
    duration = Column(Float, nullable=True)  # Event duration in seconds
    load_time = Column(Float, nullable=True)  # Page load time in seconds
    response_time = Column(Float, nullable=True)  # API response time in seconds
    
    # Demographics (for user events)
    demographics = Column(Text, nullable=True)  # JSON string of demographic data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<Analytics(id={self.id}, event_type={self.event_type}, event={self.event}, user_id={self.user_id})>"
    
    def to_dict(self) -> dict:
        """Convert analytics to dictionary."""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "category": self.category,
            "event": self.event,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": self.session_id,
            "properties": self.properties,
            "context": self.context,
            "user_agent": self.user_agent,
            "device": self.device,
            "browser": self.browser,
            "os": self.os,
            "ip_address": self.ip_address,
            "country": self.country,
            "region": self.region,
            "city": self.city,
            "referrer": self.referrer,
            "referrer_domain": self.referrer_domain,
            "duration": self.duration,
            "load_time": self.load_time,
            "response_time": self.response_time,
            "demographics": self.demographics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
