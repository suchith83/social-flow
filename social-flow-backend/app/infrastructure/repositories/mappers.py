"""
Mapper utilities for converting between domain entities and database models.

These mappers handle the translation between:
- Domain entities (rich business logic)
- SQLAlchemy ORM models (persistence)
"""

from typing import Optional
from uuid import UUID

from app.models.user import User as UserModel
from app.domain.entities.user import UserEntity
from app.domain.value_objects import Email, Username, UserRole
from app.models.video import Video as VideoModel, VideoStatus as DBVideoStatus, VideoVisibility as DBVideoVisibility
from app.domain.entities.video import VideoEntity
from app.domain.value_objects import (
    VideoStatus,
    VideoVisibility,
    VideoMetadata,
    StorageLocation,
    EngagementMetrics,
    PostVisibility,
)
from app.models.social import Post as PostModel
from app.domain.entities.post import PostEntity


class UserMapper:
    """Maps between UserEntity and User SQLAlchemy model."""
    
    @staticmethod
    def to_entity(model: UserModel) -> UserEntity:
        """Convert SQLAlchemy model to domain entity."""
        entity = UserEntity(
            username=Username(model.username),
            email=Email(model.email),
            password_hash=model.password_hash,
            display_name=model.display_name,
            id=model.id,
            role=UserRole(model.role) if hasattr(model, 'role') else UserRole.USER,
        )
        
        # Set internal state (bypassing business logic)
        entity._bio = model.bio
        entity._avatar_url = model.avatar_url
        entity._website = model.website
        entity._location = model.location
        
        entity._is_active = model.is_active
        entity._is_verified = model.is_verified
        entity._is_banned = model.is_banned
        entity._is_suspended = model.is_suspended
        
        entity._ban_reason = model.ban_reason
        entity._banned_at = model.banned_at
        entity._suspension_reason = model.suspension_reason
        entity._suspended_at = model.suspended_at
        entity._suspension_ends_at = model.suspension_ends_at
        
        entity._followers_count = model.followers_count
        entity._following_count = model.following_count
        entity._posts_count = model.posts_count
        entity._videos_count = model.videos_count
        entity._total_views = model.total_views
        entity._total_likes = model.total_likes
        
        entity._email_notifications = model.email_notifications
        entity._push_notifications = model.push_notifications
        entity._privacy_level = model.privacy_level
        
        entity._last_login_at = model.last_login_at
        entity._created_at = model.created_at
        entity._updated_at = model.updated_at
        
        return entity
    
    @staticmethod
    def to_model(entity: UserEntity, model: Optional[UserModel] = None) -> UserModel:
        """Convert domain entity to SQLAlchemy model."""
        if model is None:
            model = UserModel()
            model.id = entity.id
        
        model.username = str(entity.username)
        model.email = str(entity.email)
        model.password_hash = entity.password_hash
        model.display_name = entity.display_name
        
        model.bio = entity.bio
        model.avatar_url = entity.avatar_url
        model.website = entity.website
        model.location = entity.location
        
        model.is_active = entity.is_active
        model.is_verified = entity.is_verified
        model.is_banned = entity.is_banned
        model.is_suspended = entity.is_suspended
        
        model.ban_reason = entity._ban_reason
        model.banned_at = entity._banned_at
        model.suspension_reason = entity._suspension_reason
        model.suspended_at = entity._suspended_at
        model.suspension_ends_at = entity._suspension_ends_at
        
        model.followers_count = entity.followers_count
        model.following_count = entity.following_count
        model.posts_count = entity.posts_count
        model.videos_count = entity.videos_count
        model.total_views = entity.total_views
        model.total_likes = entity.total_likes
        
        model.email_notifications = entity._email_notifications
        model.push_notifications = entity._push_notifications
        model.privacy_level = entity._privacy_level
        
        model.last_login_at = entity.last_login_at
        
        return model


class VideoMapper:
    """Maps between VideoEntity and Video SQLAlchemy model."""
    
    @staticmethod
    def to_entity(model: VideoModel) -> VideoEntity:
        """Convert SQLAlchemy model to domain entity."""
        # Create metadata
        metadata = VideoMetadata(
            duration=model.duration or 0.0,
            resolution=model.resolution or "0x0",
            bitrate=model.bitrate or 0,
            codec=model.codec or "unknown",
            file_size=model.file_size,
        )
        
        # Create storage location
        storage_location = StorageLocation(
            bucket=model.s3_bucket,
            key=model.s3_key,
        )
        
        # Create entity
        entity = VideoEntity(
            title=model.title,
            owner_id=model.owner_id,
            storage_location=storage_location,
            metadata=metadata,
            id=model.id,
        )
        
        # Set internal state
        entity._description = model.description
        entity._tags = model.tags.split(',') if model.tags else []
        
        entity._hls_url = model.hls_url
        entity._dash_url = model.dash_url
        entity._thumbnail_url = model.thumbnail_url
        entity._preview_url = model.preview_url
        entity._available_qualities = model.available_qualities.split(',') if model.available_qualities else []
        
        entity._status = VideoStatus(model.status.value)
        entity._visibility = VideoVisibility(model.visibility.value)
        
        entity._is_approved = model.is_approved
        entity._is_flagged = model.is_flagged
        entity._is_rejected = model.is_rejected
        entity._flag_reason = model.flag_reason
        entity._rejection_reason = model.rejection_reason
        entity._approved_at = model.approved_at
        entity._flagged_at = model.flagged_at
        entity._rejected_at = model.rejected_at
        
        entity._engagement = EngagementMetrics(
            views=model.views_count,
            likes=model.likes_count,
            dislikes=model.dislikes_count,
            comments=model.comments_count,
            shares=model.shares_count,
        )
        
        entity._total_watch_time = model.total_watch_time
        entity._average_watch_time = model.average_watch_time
        entity._retention_rate = model.retention_rate
        
        entity._is_monetized = model.is_monetized
        entity._ad_revenue = model.ad_revenue
        
        entity._processing_started_at = model.processing_started_at
        entity._processing_completed_at = model.processing_completed_at
        entity._processing_error = model.processing_error
        
        entity._created_at = model.created_at
        entity._updated_at = model.updated_at
        
        return entity
    
    @staticmethod
    def to_model(entity: VideoEntity, model: Optional[VideoModel] = None) -> VideoModel:
        """Convert domain entity to SQLAlchemy model."""
        if model is None:
            model = VideoModel()
            model.id = entity.id
        
        model.title = entity.title
        model.description = entity.description
        model.tags = ','.join(entity.tags) if entity.tags else None
        
        model.filename = entity.storage_location.key.split('/')[-1]
        model.file_size = entity.metadata.file_size
        model.duration = entity.metadata.duration
        model.resolution = entity.metadata.resolution
        model.bitrate = entity.metadata.bitrate
        model.codec = entity.metadata.codec
        
        model.s3_key = entity.storage_location.key
        model.s3_bucket = entity.storage_location.bucket
        model.thumbnail_url = entity.thumbnail_url
        model.preview_url = entity._preview_url
        
        model.hls_url = entity.hls_url
        model.dash_url = entity._dash_url
        model.streaming_url = entity.hls_url  # Use HLS as default streaming URL
        model.available_qualities = ','.join(entity._available_qualities) if entity._available_qualities else None
        
        model.status = DBVideoStatus(entity.status.value)
        model.visibility = DBVideoVisibility(entity.visibility.value)
        
        model.is_approved = entity.is_approved
        model.is_flagged = entity.is_flagged
        model.is_rejected = entity.is_rejected
        model.approved_at = entity._approved_at
        model.approved_by = None  # TODO: Track who approved
        model.flagged_at = entity._flagged_at
        model.flagged_by = None  # TODO: Track who flagged
        model.flag_reason = entity._flag_reason
        model.rejected_at = entity._rejected_at
        model.rejected_by = None  # TODO: Track who rejected
        model.rejection_reason = entity._rejection_reason
        
        model.views_count = entity.engagement.views
        model.likes_count = entity.engagement.likes
        model.dislikes_count = entity.engagement.dislikes
        model.comments_count = entity.engagement.comments
        model.shares_count = entity.engagement.shares
        
        model.total_watch_time = entity._total_watch_time
        model.average_watch_time = entity._average_watch_time
        model.retention_rate = entity._retention_rate
        
        model.is_monetized = entity.is_monetized
        model.ad_revenue = entity.ad_revenue
        
        model.processing_started_at = entity._processing_started_at
        model.processing_completed_at = entity._processing_completed_at
        model.processing_error = entity._processing_error
        
        model.owner_id = entity.owner_id
        
        return model


class PostMapper:
    """Maps between PostEntity and Post SQLAlchemy model."""
    
    @staticmethod
    def to_entity(model: PostModel) -> PostEntity:
        """Convert SQLAlchemy model to domain entity."""
        entity = PostEntity(
            content=model.content,
            owner_id=model.owner_id,
            id=model.id,
        )
        
        # Set internal state
        # Note: Post model doesn't have image_urls/video_ids as separate fields
        # Extract from media_url if present
        if model.media_url and model.media_type == 'image':
            entity._image_urls = [model.media_url]
        else:
            entity._image_urls = []
        entity._video_ids = []
        
        # Note: Post model doesn't have visibility field - default to PUBLIC
        entity._visibility = PostVisibility.PUBLIC
        
        # Map is_approved to is_published
        entity._is_published = model.is_approved
        entity._published_at = model.approved_at
        
        # Map moderation fields
        entity._is_flagged = model.is_flagged
        entity._is_removed = model.is_rejected  # Map is_rejected to is_removed
        entity._flag_reason = model.flag_reason
        entity._flagged_at = model.flagged_at
        entity._removed_at = model.rejected_at
        entity._removal_reason = model.rejection_reason
        
        entity._engagement = EngagementMetrics(
            views=model.views_count,
            likes=model.likes_count,
            dislikes=getattr(model, 'dislikes_count', 0),
            comments=model.comments_count,
            shares=model.shares_count,
        )
        
        entity._created_at = model.created_at
        entity._updated_at = model.updated_at
        
        return entity
    
    @staticmethod
    def to_model(entity: PostEntity, model: Optional[PostModel] = None) -> PostModel:
        """Convert domain entity to SQLAlchemy model."""
        if model is None:
            model = PostModel()
            model.id = entity.id
        
        model.content = entity.content
        # Note: Post model doesn't have image_urls/video_ids fields yet
        # These should be in media_url field or a separate media table
        if entity.image_urls:
            model.media_url = entity.image_urls[0] if entity.image_urls else None
            model.media_type = 'image'
        
        # Note: Post model doesn't have visibility field - it's always public for now
        # model.visibility = entity.visibility.value
        
        # The model uses is_approved instead of is_published
        model.is_approved = entity.is_published
        model.approved_at = entity.published_at
        
        model.is_flagged = entity.is_flagged
        # The model doesn't have is_removed, only is_rejected
        model.is_rejected = entity.is_removed
        model.flag_reason = entity._flag_reason
        model.flagged_at = entity._flagged_at
        model.rejected_at = entity._removed_at
        model.rejection_reason = entity._removal_reason
        
        model.views_count = entity.engagement.views
        model.likes_count = entity.engagement.likes
        model.comments_count = entity.engagement.comments
        model.shares_count = entity.engagement.shares
        
        model.owner_id = entity.owner_id
        
        return model


