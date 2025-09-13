# Pydantic schemas for requests/responses
"""
Pydantic schemas for Mobile Optimized APIs.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class OptimizedItem(BaseModel):
    id: int
    title: str
    short_description: str
    thumbnail_url: Optional[str] = None


class OptimizedItemListResponse(BaseModel):
    items: List[OptimizedItem]
    next_offset: Optional[int] = Field(None, description="Next offset for pagination")
    has_more: bool = Field(False, description="Whether more data exists")


class DeltaSyncRequest(BaseModel):
    last_sync_timestamp: str = Field(..., description="ISO timestamp of last successful sync")


class DeltaSyncResponse(BaseModel):
    updated_items: List[OptimizedItem]
    deleted_item_ids: List[int]
    server_timestamp: str
