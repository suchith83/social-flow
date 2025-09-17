"""
Data models for analytics events and aggregated metrics
"""

from pydantic import BaseModel, Field
from typing import Optional
import datetime


class AnalyticsEvent(BaseModel):
    event_id: str
    user_id: Optional[str]
    video_id: str
    event_type: str = Field(..., description="view, like, comment, share, etc.")
    timestamp: datetime.datetime
    metadata: dict = {}


class AggregatedMetrics(BaseModel):
    video_id: str
    total_views: int
    total_likes: int
    total_comments: int
    total_watch_time: float  # in seconds
    last_updated: datetime.datetime
