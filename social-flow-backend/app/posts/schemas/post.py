"""
Post Pydantic schemas for request/response validation.

This module defines all Pydantic schemas related to posts.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class PostBase(BaseModel):
    """Base schema for posts."""
    content: str = Field(..., min_length=1, max_length=2000, description="Post content")
    media_url: Optional[str] = Field(None, max_length=500, description="Media URL")
    media_type: Optional[str] = Field(None, max_length=50, description="Media type (image, video, gif)")


class PostCreate(PostBase):
    """Schema for creating a post."""
    pass


class PostUpdate(BaseModel):
    """Schema for updating a post."""
    content: Optional[str] = Field(None, min_length=1, max_length=2000)
    media_url: Optional[str] = Field(None, max_length=500)
    media_type: Optional[str] = Field(None, max_length=50)


class RepostCreate(BaseModel):
    """Schema for reposting."""
    original_post_id: UUID
    reason: Optional[str] = Field(None, max_length=500, description="Optional reason for repost")


class UserMinimal(BaseModel):
    """Minimal user information for post responses."""
    id: UUID
    username: str
    display_name: str
    avatar_url: Optional[str]
    is_verified: bool
    
    class Config:
        from_attributes = True


class PostResponse(PostBase):
    """Schema for post responses."""
    id: UUID
    owner_id: UUID
    owner: Optional[UserMinimal]
    
    # Engagement metrics
    likes_count: int
    reposts_count: int
    comments_count: int
    shares_count: int
    views_count: int
    
    # Repost information
    is_repost: bool
    original_post_id: Optional[UUID]
    repost_reason: Optional[str]
    original_post: Optional['PostResponse']
    
    # Hashtags and mentions
    hashtags: Optional[str]  # JSON string
    mentions: Optional[str]  # JSON string
    
    # Moderation
    is_approved: bool
    is_flagged: bool
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
    
    @validator('hashtags', 'mentions', pre=True)
    def parse_json_fields(cls, value):
        """Parse JSON string fields."""
        if isinstance(value, str):
            import json
            try:
                return json.loads(value)
            except:
                return []
        return value


class PostListResponse(BaseModel):
    """Schema for list of posts."""
    posts: List[PostResponse]
    total: int
    skip: int
    limit: int


class FeedResponse(BaseModel):
    """Schema for feed response."""
    posts: List[PostResponse]
    algorithm: str
    has_more: bool


class LikeRequest(BaseModel):
    """Schema for like request."""
    post_id: UUID


class LikeResponse(BaseModel):
    """Schema for like response."""
    post_id: UUID
    user_id: UUID
    created_at: datetime
    
    class Config:
        from_attributes = True


# Update forward references
PostResponse.model_rebuild()
