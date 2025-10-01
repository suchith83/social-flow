"""
Comment Pydantic schemas for request/response validation.

This module defines all Pydantic schemas related to comments.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class CommentBase(BaseModel):
    """Base schema for comments."""
    content: str = Field(..., min_length=1, max_length=1000, description="Comment content")


class CommentCreate(CommentBase):
    """Schema for creating a comment."""
    post_id: Optional[UUID] = Field(None, description="Post ID this comment belongs to")
    video_id: Optional[UUID] = Field(None, description="Video ID this comment belongs to")
    parent_comment_id: Optional[UUID] = Field(None, description="Parent comment ID for replies")

    @field_validator('post_id', 'video_id')
    @classmethod
    def validate_target(cls, v, info):
        """Ensure comment has exactly one target (post or video)."""
        values = info.data
        if info.field_name == 'post_id' and v is not None:
            if values.get('video_id') is not None:
                raise ValueError("Comment cannot belong to both a post and a video")
        elif info.field_name == 'video_id' and v is not None:
            if values.get('post_id') is not None:
                raise ValueError("Comment cannot belong to both a post and a video")

        # At least one target must be provided
        if not v and not values.get('post_id', values.get('video_id')):
            if info.field_name == 'post_id':
                # This will be checked on video_id as well, but that's ok
                pass
            else:
                raise ValueError("Comment must belong to either a post or a video")

        return v


class CommentUpdate(BaseModel):
    """Schema for updating a comment."""
    content: Optional[str] = Field(None, min_length=1, max_length=1000)


class UserMinimal(BaseModel):
    """Minimal user information for comment responses."""
    id: UUID
    username: str
    display_name: str
    avatar_url: Optional[str]
    is_verified: bool

    class Config:
        from_attributes = True


class CommentResponse(CommentBase):
    """Schema for comment responses."""
    id: UUID
    owner_id: UUID
    owner: Optional[UserMinimal]

    # Engagement metrics
    likes_count: int
    replies_count: int

    # Reply information
    is_reply: bool
    parent_comment_id: Optional[UUID]

    # Hashtags and mentions
    hashtags: Optional[str]  # JSON string
    mentions: Optional[str]  # JSON string

    # Moderation
    is_approved: bool
    is_flagged: bool
    is_rejected: bool

    # Foreign keys
    post_id: Optional[UUID]
    video_id: Optional[UUID]

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator('hashtags', 'mentions', mode='before')
    @classmethod
    def parse_json_fields(cls, v):
        """Parse JSON string fields."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except Exception:
                return []
        return v


class CommentListResponse(BaseModel):
    """Schema for list of comments."""
    comments: List[CommentResponse]
    total: int
    skip: int
    limit: int


class CommentRepliesResponse(BaseModel):
    """Schema for comment replies."""
    replies: List[CommentResponse]
    total: int
    skip: int
    limit: int


class LikeRequest(BaseModel):
    """Schema for like request."""
    comment_id: UUID


class LikeResponse(BaseModel):
    """Schema for like response."""
    comment_id: UUID
    user_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True