"""
Social features Pydantic schemas for API request/response validation.

This module provides schemas for social interactions including:
- Posts (text/image posts with hashtags and mentions)
- Comments (threaded comments on posts and videos)
- Likes (universal likes for posts, videos, comments)
- Follows (user following relationships)
- Saves (bookmarked content)
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import Field, field_validator

from app.schemas.base import BaseDBSchema, BaseSchema


# ==================== POST SCHEMAS ====================

class PostStatus(str):
    """Post status enumeration."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    FLAGGED = "flagged"
    REMOVED = "removed"


class PostVisibility(str):
    """Post visibility settings."""
    PUBLIC = "public"
    FOLLOWERS_ONLY = "followers_only"
    PRIVATE = "private"


class PostBase(BaseSchema):
    """Base post schema."""
    
    content: str = Field(..., min_length=1, max_length=5000, description="Post content")
    images: list[str] = Field(default_factory=list, max_length=10, description="Image URLs")
    visibility: str = Field(default="public", description="Post visibility")
    allow_comments: bool = Field(default=True, description="Allow comments")
    allow_likes: bool = Field(default=True, description="Allow likes")


class PostCreate(PostBase):
    """Create new post."""
    
    repost_of_id: Optional[UUID] = Field(None, description="ID of post being reposted")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str, info) -> str:
        """Validate content is provided for original posts."""
        # If it's a repost, content is optional
        if info.data.get('repost_of_id') is None and not v.strip():
            raise ValueError('Content is required for original posts')
        return v


class PostUpdate(BaseSchema):
    """Update existing post."""
    
    content: Optional[str] = Field(None, min_length=1, max_length=5000)
    images: Optional[list[str]] = Field(None, max_length=10)
    visibility: Optional[str] = None
    allow_comments: Optional[bool] = None
    allow_likes: Optional[bool] = None


class PostResponse(BaseDBSchema):
    """Post response schema."""
    
    owner_id: UUID
    content: str
    media_urls: list[str] = []
    
    # Repost info
    original_post_id: Optional[UUID] = None
    is_repost: bool = False
    
    # Extracted data
    hashtags: list[str] = []
    mentions: list[str] = []
    
    # Settings
    visibility: str
    
    # Stats
    like_count: int = 0
    comment_count: int = 0
    repost_count: int = 0
    view_count: int = 0
    
    # Moderation
    is_flagged: bool = False
    
    model_config = {"from_attributes": True}


class PostDetailResponse(PostResponse):
    """Detailed post response with related data."""
    
    # Original post (if repost)
    original_post: Optional[PostResponse] = None
    
    # User interaction flags (requires authenticated user)
    is_liked: bool = False
    is_saved: bool = False


# ==================== COMMENT SCHEMAS ====================

class CommentStatus(str):
    """Comment status enumeration."""
    PUBLISHED = "published"
    FLAGGED = "flagged"
    REMOVED = "removed"
    HIDDEN = "hidden"


class CommentBase(BaseSchema):
    """Base comment schema."""
    
    content: str = Field(..., min_length=1, max_length=2000, description="Comment content")


class CommentCreate(CommentBase):
    """Create new comment."""
    
    post_id: Optional[UUID] = Field(None, description="Post ID (for post comments)")
    video_id: Optional[UUID] = Field(None, description="Video ID (for video comments)")
    parent_comment_id: Optional[UUID] = Field(None, description="Parent comment ID (for replies)")
    
    @field_validator('post_id')
    @classmethod
    def validate_target(cls, v: Optional[UUID], info) -> Optional[UUID]:
        """Validate that comment has a target (post or video)."""
        video_id = info.data.get('video_id')
        if v is None and video_id is None:
            raise ValueError('Comment must target either a post or a video')
        if v is not None and video_id is not None:
            raise ValueError('Comment cannot target both a post and a video')
        return v


class CommentUpdate(BaseSchema):
    """Update existing comment."""
    
    content: Optional[str] = Field(None, min_length=1, max_length=2000)


class CommentResponse(BaseDBSchema):
    """Comment response schema."""
    
    user_id: UUID
    content: str
    
    # Target
    post_id: Optional[UUID] = None
    video_id: Optional[UUID] = None
    parent_comment_id: Optional[UUID] = None
    
    # Settings
    is_edited: bool = False
    
    # Stats
    like_count: int = 0
    reply_count: int = 0
    
    # Moderation
    is_flagged: bool = False
    
    # Timestamps
    edited_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class CommentDetailResponse(CommentResponse):
    """Detailed comment with replies."""
    
    # User interaction
    is_liked: bool = False
    
    # Nested replies (limited depth)
    replies: list[CommentResponse] = []


# ==================== LIKE SCHEMAS ====================

class LikeCreate(BaseSchema):
    """Create a like."""
    
    user_id: UUID = Field(..., description="User ID")
    post_id: Optional[UUID] = Field(None, description="Post ID")
    video_id: Optional[UUID] = Field(None, description="Video ID")
    comment_id: Optional[UUID] = Field(None, description="Comment ID")
    
    @field_validator('post_id')
    @classmethod
    def validate_single_target(cls, v: Optional[UUID], info) -> Optional[UUID]:
        """Validate that like targets exactly one resource."""
        targets = [
            v,
            info.data.get('video_id'),
            info.data.get('comment_id')
        ]
        non_none = [t for t in targets if t is not None]
        
        if len(non_none) == 0:
            raise ValueError('Like must target a post, video, or comment')
        if len(non_none) > 1:
            raise ValueError('Like can only target one resource')
        return v


class LikeResponse(BaseDBSchema):
    """Like response schema."""
    
    user_id: UUID
    post_id: Optional[UUID] = None
    video_id: Optional[UUID] = None
    comment_id: Optional[UUID] = None


# ==================== FOLLOW SCHEMAS ====================

class FollowCreate(BaseSchema):
    """Create a follow relationship."""
    
    following_id: UUID = Field(..., description="ID of user to follow")


class FollowResponse(BaseDBSchema):
    """Follow relationship response."""
    
    follower_id: UUID
    following_id: UUID
    is_accepted: bool = True
    
    # For mutual follows
    is_mutual: bool = False


class FollowerResponse(BaseSchema):
    """Follower/following user info."""
    
    user_id: UUID
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_verified: bool = False
    followers_count: int = 0
    
    # Relationship status
    is_following: bool = False
    is_followed_by: bool = False
    is_mutual: bool = False


# ==================== SAVE SCHEMAS ====================

class SaveCreate(BaseSchema):
    """Save/bookmark content."""
    
    post_id: Optional[UUID] = Field(None, description="Post ID")
    video_id: Optional[UUID] = Field(None, description="Video ID")
    
    @field_validator('post_id')
    @classmethod
    def validate_single_target(cls, v: Optional[UUID], info) -> Optional[UUID]:
        """Validate that save targets exactly one resource."""
        video_id = info.data.get('video_id')
        
        if v is None and video_id is None:
            raise ValueError('Save must target a post or video')
        if v is not None and video_id is not None:
            raise ValueError('Save can only target one resource')
        return v


class SaveResponse(BaseDBSchema):
    """Saved content response."""
    
    user_id: UUID
    post_id: Optional[UUID] = None
    video_id: Optional[UUID] = None


# ==================== LIST AND FILTER SCHEMAS ====================

class PostListFilters(BaseSchema):
    """Filters for post listing."""
    
    user_id: Optional[UUID] = None
    status: Optional[str] = None
    visibility: Optional[str] = None
    hashtag: Optional[str] = None
    search: Optional[str] = Field(None, description="Search in content")
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class CommentListFilters(BaseSchema):
    """Filters for comment listing."""
    
    user_id: Optional[UUID] = None
    post_id: Optional[UUID] = None
    video_id: Optional[UUID] = None
    parent_comment_id: Optional[UUID] = None
    status: Optional[str] = None
    created_after: Optional[datetime] = None


class FeedFilters(BaseSchema):
    """Filters for personalized feed."""
    
    feed_type: str = Field(
        default="following",
        description="Feed type: following, trending, latest"
    )
    content_type: Optional[str] = Field(
        None,
        description="Content type: posts, videos, all"
    )
