"""
Post schemas for request/response validation.

This module contains Pydantic schemas for post management.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class PostBase(BaseModel):
    """Base post schema."""
    content: str = Field(..., min_length=1, max_length=5000)
    media_url: Optional[str] = None
    media_type: Optional[str] = Field(None, pattern='^(image|video|gif)$')
    hashtags: Optional[str] = None
    mentions: Optional[str] = None
    visibility: Optional[str] = Field('public', pattern='^(public|private|followers)$')


class PostCreate(PostBase):
    """Schema for creating a new post."""
    is_repost: Optional[bool] = False
    original_post_id: Optional[str] = None
    repost_reason: Optional[str] = None
    
    @validator('original_post_id')
    def validate_repost(cls, v, values):
        """Validate repost data consistency."""
        if values.get('is_repost') and not v:
            raise ValueError('original_post_id required when is_repost is True')
        return v


class PostUpdate(BaseModel):
    """Schema for updating a post."""
    content: Optional[str] = Field(None, min_length=1, max_length=5000)
    media_url: Optional[str] = None
    media_type: Optional[str] = Field(None, pattern='^(image|video|gif)$')
    hashtags: Optional[str] = None
    mentions: Optional[str] = None


class RepostCreate(BaseModel):
    """Schema for creating a repost."""
    original_post_id: str
    repost_reason: Optional[str] = Field(None, max_length=500)


class PostResponse(PostBase):
    """Schema for post response."""
    id: str
    owner_id: str
    likes_count: int
    reposts_count: int
    comments_count: int
    shares_count: int
    views_count: int
    is_approved: bool
    is_flagged: bool
    is_rejected: bool
    is_repost: bool
    original_post_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PostListResponse(BaseModel):
    """Schema for paginated post list response."""
    posts: List[PostResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class CommentCreate(BaseModel):
    """Schema for creating a comment."""
    content: str = Field(..., min_length=1, max_length=2000)
    parent_id: Optional[str] = None


class CommentUpdate(BaseModel):
    """Schema for updating a comment."""
    content: str = Field(..., min_length=1, max_length=2000)


class CommentResponse(BaseModel):
    """Schema for comment response."""
    id: str
    content: str
    user_id: str
    post_id: str
    parent_id: Optional[str]
    likes_count: int
    replies_count: int
    is_flagged: bool
    is_deleted: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class LikeResponse(BaseModel):
    """Schema for like response."""
    id: str
    user_id: str
    post_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True
