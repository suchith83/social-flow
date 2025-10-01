"""
Like Pydantic schemas for request/response validation.

This module defines all Pydantic schemas related to likes.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class LikeBase(BaseModel):
    """Base schema for likes."""
    like_type: str = Field(..., pattern="^(post|video|comment)$", description="Type of entity being liked")
    is_like: bool = Field(True, description="True for like, False for dislike")


class LikeCreate(LikeBase):
    """Schema for creating a like."""
    post_id: Optional[UUID] = Field(None, description="Post ID being liked")
    video_id: Optional[UUID] = Field(None, description="Video ID being liked")
    comment_id: Optional[UUID] = Field(None, description="Comment ID being liked")

    @field_validator('post_id', 'video_id', 'comment_id')
    @classmethod
    def validate_target(cls, v, info):
        """Ensure like has exactly one target."""
        values = info.data
        provided_targets = [t for t in ['post_id', 'video_id', 'comment_id'] if values.get(t) is not None]

        if info.field_name in provided_targets and len(provided_targets) > 1:
            raise ValueError("Like can only target one entity at a time")

        return v

    @field_validator('like_type')
    @classmethod
    def validate_like_type(cls, v, info):
        """Ensure like_type matches the provided target."""
        values = info.data
        if v == "post" and not values.get('post_id'):
            raise ValueError("like_type 'post' requires post_id")
        elif v == "video" and not values.get('video_id'):
            raise ValueError("like_type 'video' requires video_id")
        elif v == "comment" and not values.get('comment_id'):
            raise ValueError("like_type 'comment' requires comment_id")

        return v


class LikeUpdate(LikeBase):
    """Schema for updating a like."""
    post_id: Optional[UUID] = Field(None, description="Post ID")
    video_id: Optional[UUID] = Field(None, description="Video ID")
    comment_id: Optional[UUID] = Field(None, description="Comment ID")

    @field_validator('post_id', 'video_id', 'comment_id')
    @classmethod
    def validate_target(cls, v, info):
        """Ensure update has exactly one target."""
        values = info.data
        targets = ['post_id', 'video_id', 'comment_id']
        provided_targets = [t for t in targets if values.get(t) is not None]

        if info.field_name in provided_targets and len(provided_targets) > 1:
            raise ValueError("Like update can only target one entity at a time")

        return v


class LikeResponse(LikeBase):
    """Schema for like responses."""
    id: UUID
    user_id: UUID
    post_id: Optional[UUID]
    video_id: Optional[UUID]
    comment_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LikeListResponse(BaseModel):
    """Schema for list of likes."""
    likes: List[LikeResponse]
    total: int
    skip: int
    limit: int


class LikeCheckResponse(BaseModel):
    """Schema for like check response."""
    is_liked: bool
    like_id: Optional[UUID]
    is_like: Optional[bool]  # True for like, False for dislike