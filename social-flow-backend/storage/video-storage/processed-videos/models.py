"""
Data models for processed video pipeline
"""

from pydantic import BaseModel
from typing import List, Dict
import datetime


class ProcessedVideo(BaseModel):
    video_id: str
    resolutions: List[str]
    thumbnails: List[str]
    metadata: Dict
    quality: Dict
    storage_urls: List[str]
    processed_at: datetime.datetime
