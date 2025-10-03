"""
Social models for posts, comments, likes, and follows.

This module defines models for social features including:
- Posts (text, media, reposts)
- Comments (nested threading)
- Likes (videos, posts, comments)
- Follows (user relationships)
- Social engagement tracking
"""

from __future__ import annotations

from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum as SQLEnum,
    Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
)
from sqlalchemy.orm import Mapped, relationship

from app.models.base import CommonBase
from app.models.types import ARRAY, JSONB, UUID

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.video import Video


class PostVisibility(str, PyEnum):
    """Post visibility level."""
    PUBLIC = "public"
    FOLLOWERS = "followers"
    MENTIONED = "mentioned"
    PRIVATE = "private"


class MediaType(str, PyEnum):
    """Media type for posts."""
    IMAGE = "image"
    VIDEO = "video"
    GIF = "gif"
    LINK = "link"


class Post(CommonBase):
    """
    Post model for social media posts.
    
    Supports:
    - Text posts
    - Media posts (images, videos, GIFs)
    - Link sharing
    - Reposts/quotes
    - Hashtags and mentions
    - Engagement metrics
    """
    
    __tablename__ = "posts"
    
    # ==================== Content ====================
    content = Column(
        Text,
        nullable=True,
        doc="Post content/text (required unless is_repost)"
    )
    
    content_html = Column(
        Text,
        nullable=True,
        doc="Rendered HTML content with links and mentions"
    )
    
    # ==================== Media ====================
    media_type = Column(
        SQLEnum(MediaType),
        nullable=True,
        doc="Type of media attached"
    )
    
    media_urls = Column(
        ARRAY(String(500)),
        default=[],
        nullable=False,
        doc="Array of media URLs (supports multiple images)"
    )
    
    media_metadata = Column(
        JSONB,
        default={},
        nullable=False,
        doc="Media metadata (dimensions, duration, etc.)"
    )
    
    # ==================== Social ====================
    hashtags = Column(
        ARRAY(String(100)),
        default=[],
        nullable=False,
        doc="Array of hashtags (without # prefix)"
    )
    
    mentions = Column(
        ARRAY(UUID(as_uuid=True)),
        default=[],
        nullable=False,
        doc="Array of mentioned user IDs"
    )
    
    # ==================== Owner ====================
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Post author user ID"
    )
    
    # ==================== Repost ====================
    is_repost = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether this is a repost"
    )
    
    original_post_id = Column(
        UUID(as_uuid=True),
        ForeignKey('posts.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Original post ID if this is a repost"
    )
    
    repost_comment = Column(
        Text,
        nullable=True,
        doc="Comment/quote added to repost"
    )
    
    # ==================== Visibility ====================
    visibility = Column(
        SQLEnum(PostVisibility),
        default=PostVisibility.PUBLIC,
        nullable=False,
        index=True,
        doc="Post visibility level"
    )
    
    # ==================== Engagement Metrics ====================
    view_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        index=True,
        doc="Total number of views"
    )
    
    like_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        index=True,
        doc="Total number of likes"
    )
    
    comment_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of comments"
    )
    
    repost_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of reposts"
    )
    
    share_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of shares (external)"
    )
    
    save_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total number of saves/bookmarks"
    )
    
    # ==================== Engagement Rate ====================
    engagement_rate = Column(
        Float,
        default=0.0,
        nullable=False,
        index=True,
        doc="Engagement rate (interactions / views)"
    )
    
    # ==================== Moderation ====================
    is_flagged = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether post has been flagged"
    )
    
    flag_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of times post has been flagged"
    )
    
    moderation_notes = Column(
        Text,
        nullable=True,
        doc="Moderator notes"
    )
    
    moderated_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Moderation timestamp"
    )
    
    moderated_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="Moderator user ID"
    )
    
    # ==================== Relationships ====================
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="posts",
        foreign_keys=[owner_id]
    )
    
    original_post: Mapped["Post"] = relationship(
        "Post",
        remote_side="Post.id",
        foreign_keys=[original_post_id],
        backref="reposts"
    )
    
    comments: Mapped[list["Comment"]] = relationship(
        "Comment",
        back_populates="post",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    likes: Mapped[list["Like"]] = relationship(
        "Like",
        back_populates="post",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    moderator = relationship(
        "User",
        foreign_keys=[moderated_by_id]
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_post_owner_created', 'owner_id', 'created_at'),
        Index('idx_post_visibility_created', 'visibility', 'created_at'),
        Index('idx_post_engagement', 'engagement_rate', 'created_at'),
        Index('idx_post_hashtags', 'hashtags', postgresql_using='gin'),
        Index('idx_post_mentions', 'mentions', postgresql_using='gin'),
        
        # Full-text search
        Index(
            'idx_post_content_search',
            'content',
            postgresql_using='gin',
            postgresql_ops={'content': 'gin_trgm_ops'}
        ),
        
        # Partition by created_at
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<Post(id={self.id}, owner_id={self.owner_id})>"


class Comment(CommonBase):
    """
    Comment model for posts and videos.
    
    Supports:
    - Comments on posts
    - Comments on videos
    - Nested/threaded comments (replies)
    - Mentions and likes
    """
    
    __tablename__ = "comments"
    
    # ==================== Content ====================
    content = Column(
        Text,
        nullable=False,
        doc="Comment content"
    )
    
    content_html = Column(
        Text,
        nullable=True,
        doc="Rendered HTML content"
    )
    
    # ==================== Author ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Comment author user ID"
    )
    
    # ==================== Target (Post or Video) ====================
    post_id = Column(
        UUID(as_uuid=True),
        ForeignKey('posts.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Post ID if comment is on a post"
    )
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Video ID if comment is on a video"
    )
    
    # ==================== Threading (Nested Comments) ====================
    parent_comment_id = Column(
        UUID(as_uuid=True),
        ForeignKey('comments.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Parent comment ID for replies"
    )
    
    reply_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of direct replies"
    )
    
    # ==================== Social ====================
    mentions = Column(
        ARRAY(UUID(as_uuid=True)),
        default=[],
        nullable=False,
        doc="Array of mentioned user IDs"
    )
    
    # ==================== Engagement ====================
    like_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        index=True,
        doc="Number of likes"
    )
    
    # ==================== Moderation ====================
    is_flagged = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether comment has been flagged"
    )
    
    moderation_notes = Column(
        Text,
        nullable=True,
        doc="Moderator notes"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        back_populates="comments",
        foreign_keys=[user_id]
    )
    
    post: Mapped["Post"] = relationship(
        "Post",
        back_populates="comments",
        foreign_keys=[post_id]
    )
    
    video: Mapped["Video"] = relationship(
        "Video",
        back_populates="comments",
        foreign_keys=[video_id]
    )
    
    parent_comment: Mapped["Comment"] = relationship(
        "Comment",
        remote_side="Comment.id",
        foreign_keys=[parent_comment_id],
        backref="replies"
    )
    
    likes: Mapped[list["Like"]] = relationship(
        "Like",
        back_populates="comment",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_comment_user_created', 'user_id', 'created_at'),
        Index('idx_comment_post_created', 'post_id', 'created_at'),
        Index('idx_comment_video_created', 'video_id', 'created_at'),
        Index('idx_comment_parent_created', 'parent_comment_id', 'created_at'),
        Index('idx_comment_like_count', 'like_count', 'created_at'),
        
        # Ensure comment is on either post or video (not both)
        # Note: This will be enforced in application logic as PostgreSQL
        # doesn't support CHECK constraints with complex logic easily
        
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<Comment(id={self.id}, user_id={self.user_id})>"


class Like(CommonBase):
    """
    Like model for posts, videos, and comments.
    
    Unified like model that can like any content type.
    """
    
    __tablename__ = "likes"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User who liked"
    )
    
    # ==================== Target (Post, Video, or Comment) ====================
    post_id = Column(
        UUID(as_uuid=True),
        ForeignKey('posts.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Post ID if liking a post"
    )
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Video ID if liking a video"
    )
    
    comment_id = Column(
        UUID(as_uuid=True),
        ForeignKey('comments.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Comment ID if liking a comment"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        back_populates="likes",
        foreign_keys=[user_id]
    )
    
    post: Mapped["Post"] = relationship(
        "Post",
        back_populates="likes",
        foreign_keys=[post_id]
    )
    
    video: Mapped["Video"] = relationship(
        "Video",
        back_populates="likes",
        foreign_keys=[video_id]
    )
    
    comment: Mapped["Comment"] = relationship(
        "Comment",
        back_populates="likes",
        foreign_keys=[comment_id]
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        # Ensure user can only like each item once
        UniqueConstraint('user_id', 'post_id', name='uq_like_user_post'),
        UniqueConstraint('user_id', 'video_id', name='uq_like_user_video'),
        UniqueConstraint('user_id', 'comment_id', name='uq_like_user_comment'),
        
        # Indexes for common queries
        Index('idx_like_user_created', 'user_id', 'created_at'),
        Index('idx_like_post_created', 'post_id', 'created_at'),
        Index('idx_like_video_created', 'video_id', 'created_at'),
        Index('idx_like_comment_created', 'comment_id', 'created_at'),
        
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<Like(id={self.id}, user_id={self.user_id})>"


class Follow(CommonBase):
    """
    Follow model for user-to-user relationships.
    
    Represents a follower/following relationship between two users.
    """
    
    __tablename__ = "follows"
    
    # ==================== Follower (User who follows) ====================
    follower_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User ID of follower"
    )
    
    # ==================== Following (User being followed) ====================
    following_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User ID being followed"
    )
    
    # ==================== Notification ====================
    notified = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether following user was notified"
    )
    
    # ==================== Relationships ====================
    follower_user: Mapped["User"] = relationship(
        "User",
        foreign_keys=[follower_id],
        back_populates="following"
    )
    
    following_user: Mapped["User"] = relationship(
        "User",
        foreign_keys=[following_id],
        back_populates="followers"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        # Ensure user can only follow another user once
        UniqueConstraint('follower_id', 'following_id', name='uq_follow_follower_following'),
        
        # Indexes for common queries
        Index('idx_follow_follower_created', 'follower_id', 'created_at'),
        Index('idx_follow_following_created', 'following_id', 'created_at'),
        
        # Prevent self-follows (enforced in application logic)
    )
    
    def __repr__(self) -> str:
        return f"<Follow(follower_id={self.follower_id}, following_id={self.following_id})>"


class Save(CommonBase):
    """
    Save/Bookmark model for posts and videos.
    
    Allows users to save content for later viewing.
    """
    
    __tablename__ = "saves"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User who saved the content"
    )
    
    # ==================== Target (Post or Video) ====================
    post_id = Column(
        UUID(as_uuid=True),
        ForeignKey('posts.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Post ID if saving a post"
    )
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        doc="Video ID if saving a video"
    )
    
    # ==================== Collection ====================
    collection_name = Column(
        String(100),
        nullable=True,
        doc="Optional collection/playlist name"
    )
    
    notes = Column(
        Text,
        nullable=True,
        doc="Optional notes about why saved"
    )
    
    # ==================== Relationships ====================
    user = relationship("User", backref="saves")
    post = relationship("Post", backref="saves")
    video = relationship("Video", backref="saves")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        # Ensure user can only save each item once
        UniqueConstraint('user_id', 'post_id', name='uq_save_user_post'),
        UniqueConstraint('user_id', 'video_id', name='uq_save_user_video'),
        
        # Indexes
        Index('idx_save_user_created', 'user_id', 'created_at'),
        Index('idx_save_user_collection', 'user_id', 'collection_name'),
    )
    
    def __repr__(self) -> str:
        return f"<Save(id={self.id}, user_id={self.user_id})>"


# Export models
__all__ = [
    'Post',
    'Comment',
    'Like',
    'Follow',
    'Save',
    'PostVisibility',
    'MediaType',
]
