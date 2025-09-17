"""
Models for live streaming
"""

from pydantic import BaseModel
import datetime
from typing import Optional


class LiveStreamSession(BaseModel):
    stream_id: str
    user_id: str
    title: str
    started_at: datetime.datetime
    ended_at: Optional[datetime.datetime] = None
    status: str  # active, ended, failed
