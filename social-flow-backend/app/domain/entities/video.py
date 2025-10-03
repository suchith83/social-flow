"""
Video Domain Entity

Rich domain model for Video with business logic and invariants.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from app.domain.entities.base import AggregateRoot, DomainEvent
from app.domain.value_objects import (
    EngagementMetrics,
    StorageLocation,
    VideoMetadata,
    VideoStatus,
    VideoVisibility,
)


class VideoUploadedEvent(DomainEvent):
    """Event raised when a video is uploaded."""
    
    def __init__(self, video_id: UUID, owner_id: UUID, title: str):
        super().__init__(
            event_type="video.uploaded",
            data={
                "video_id": str(video_id),
                "owner_id": str(owner_id),
                "title": title,
            }
        )


class VideoProcessedEvent(DomainEvent):
    """Event raised when a video is successfully processed."""
    
    def __init__(self, video_id: UUID):
        super().__init__(
            event_type="video.processed",
            data={"video_id": str(video_id)}
        )


class VideoPublishedEvent(DomainEvent):
    """Event raised when a video is published."""
    
    def __init__(self, video_id: UUID, owner_id: UUID):
        super().__init__(
            event_type="video.published",
            data={
                "video_id": str(video_id),
                "owner_id": str(owner_id),
            }
        )


class VideoViewedEvent(DomainEvent):
    """Event raised when a video is viewed."""
    
    def __init__(self, video_id: UUID, viewer_id: Optional[UUID], watch_time: float):
        super().__init__(
            event_type="video.viewed",
            data={
                "video_id": str(video_id),
                "viewer_id": str(viewer_id) if viewer_id else None,
                "watch_time": watch_time,
            }
        )


class VideoEntity(AggregateRoot):
    """
    Video domain entity with business logic.
    
    Encapsulates video-related business rules and invariants.
    """
    
    def __init__(
        self,
        title: str,
        owner_id: UUID,
        storage_location: StorageLocation,
        metadata: VideoMetadata,
        id: Optional[UUID] = None,
    ):
        super().__init__(id)
        
        # Validate title
        if not title or len(title) < 3 or len(title) > 200:
            raise ValueError("Title must be 3-200 characters")
        
        self._title = title
        self._owner_id = owner_id
        self._storage_location = storage_location
        self._metadata = metadata
        
        # Basic information
        self._description: Optional[str] = None
        self._tags: List[str] = []
        
        # Streaming information
        self._hls_url: Optional[str] = None
        self._dash_url: Optional[str] = None
        self._thumbnail_url: Optional[str] = None
        self._preview_url: Optional[str] = None
        self._available_qualities: List[str] = []
        
        # Status
        self._status = VideoStatus.UPLOADING
        self._visibility = VideoVisibility.PRIVATE
        
        # Moderation
        self._is_approved = False
        self._is_flagged = False
        self._is_rejected = False
        self._flag_reason: Optional[str] = None
        self._rejection_reason: Optional[str] = None
        self._approved_at: Optional[datetime] = None
        self._flagged_at: Optional[datetime] = None
        self._rejected_at: Optional[datetime] = None
        
        # Engagement
        self._engagement = EngagementMetrics()
        
        # Watch time metrics
        self._total_watch_time = 0.0
        self._average_watch_time = 0.0
        self._retention_rate = 0.0
        
        # Monetization
        self._is_monetized = False
        self._ad_revenue = 0.0
        
        # Processing
        self._processing_started_at: Optional[datetime] = None
        self._processing_completed_at: Optional[datetime] = None
        self._processing_error: Optional[str] = None
        
        # Raise creation event
        self._raise_event(VideoUploadedEvent(self.id, owner_id, title))
    
    # Properties (getters)
    
    @property
    def title(self) -> str:
        return self._title
    
    @property
    def owner_id(self) -> UUID:
        return self._owner_id
    
    @property
    def storage_location(self) -> StorageLocation:
        return self._storage_location
    
    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata
    
    @property
    def description(self) -> Optional[str]:
        return self._description
    
    @property
    def tags(self) -> List[str]:
        return self._tags.copy()
    
    @property
    def status(self) -> VideoStatus:
        return self._status
    
    @property
    def visibility(self) -> VideoVisibility:
        return self._visibility
    
    @property
    def is_approved(self) -> bool:
        return self._is_approved
    
    @property
    def is_flagged(self) -> bool:
        return self._is_flagged
    
    @property
    def is_rejected(self) -> bool:
        return self._is_rejected
    
    @property
    def engagement(self) -> EngagementMetrics:
        return self._engagement
    
    @property
    def hls_url(self) -> Optional[str]:
        return self._hls_url
    
    @property
    def thumbnail_url(self) -> Optional[str]:
        return self._thumbnail_url
    
    @property
    def is_monetized(self) -> bool:
        return self._is_monetized
    
    @property
    def ad_revenue(self) -> float:
        return self._ad_revenue
    
    # Business logic methods
    
    def is_processing(self) -> bool:
        """Check if video is currently being processed."""
        return self._status in [VideoStatus.UPLOADING, VideoStatus.PROCESSING]
    
    def is_ready(self) -> bool:
        """Check if video is ready for viewing."""
        return self._status == VideoStatus.PROCESSED and self._is_approved
    
    def is_public(self) -> bool:
        """Check if video is publicly visible."""
        return self._visibility == VideoVisibility.PUBLIC and self.is_ready()
    
    def can_be_viewed_by(self, user_id: Optional[UUID]) -> bool:
        """Check if video can be viewed by a user."""
        # Owner can always view
        if user_id == self._owner_id:
            return True
        
        # Must be ready
        if not self.is_ready():
            return False
        
        # Check visibility
        if self._visibility == VideoVisibility.PUBLIC:
            return True
        elif self._visibility == VideoVisibility.UNLISTED:
            return True  # Anyone with link can view
        else:  # PRIVATE
            return False
    
    def can_be_edited_by(self, user_id: UUID) -> bool:
        """Check if video can be edited by a user."""
        return user_id == self._owner_id
    
    def can_be_deleted_by(self, user_id: UUID) -> bool:
        """Check if video can be deleted by a user."""
        return user_id == self._owner_id
    
    # Mutation methods
    
    def update_details(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Update video details."""
        if title is not None:
            if len(title) < 3 or len(title) > 200:
                raise ValueError("Title must be 3-200 characters")
            self._title = title
        
        if description is not None:
            if len(description) > 5000:
                raise ValueError("Description must be max 5000 characters")
            self._description = description
        
        if tags is not None:
            if len(tags) > 50:
                raise ValueError("Maximum 50 tags allowed")
            # Validate each tag
            for tag in tags:
                if len(tag) > 50:
                    raise ValueError("Each tag must be max 50 characters")
            self._tags = tags
        
        self._mark_updated()
        self._increment_version()
    
    def set_visibility(self, visibility: VideoVisibility) -> None:
        """Set video visibility."""
        old_visibility = self._visibility
        self._visibility = visibility
        
        # Raise event if making public
        if old_visibility != VideoVisibility.PUBLIC and visibility == VideoVisibility.PUBLIC:
            if self.is_ready():
                self._raise_event(VideoPublishedEvent(self.id, self._owner_id))
        
        self._mark_updated()
        self._increment_version()
    
    def start_processing(self) -> None:
        """Mark video as processing."""
        if self._status not in [VideoStatus.UPLOADING, VideoStatus.FAILED]:
            raise ValueError(f"Cannot start processing from status: {self._status}")
        
        self._status = VideoStatus.PROCESSING
        self._processing_started_at = datetime.utcnow()
        self._processing_error = None
        self._mark_updated()
        self._increment_version()
    
    def complete_processing(
        self,
        hls_url: str,
        thumbnail_url: str,
        available_qualities: List[str],
        dash_url: Optional[str] = None,
        preview_url: Optional[str] = None,
    ) -> None:
        """Mark video processing as complete."""
        if self._status != VideoStatus.PROCESSING:
            raise ValueError("Video must be in PROCESSING status")
        
        self._status = VideoStatus.PROCESSED
        self._hls_url = hls_url
        self._dash_url = dash_url
        self._thumbnail_url = thumbnail_url
        self._preview_url = preview_url
        self._available_qualities = available_qualities
        self._processing_completed_at = datetime.utcnow()
        self._processing_error = None
        
        self._mark_updated()
        self._increment_version()
        self._raise_event(VideoProcessedEvent(self.id))
    
    def fail_processing(self, error: str) -> None:
        """Mark video processing as failed."""
        if self._status != VideoStatus.PROCESSING:
            raise ValueError("Video must be in PROCESSING status")
        
        self._status = VideoStatus.FAILED
        self._processing_error = error
        self._mark_updated()
        self._increment_version()
    
    def approve(self) -> None:
        """Approve video for publishing."""
        if self._status != VideoStatus.PROCESSED:
            raise ValueError("Video must be processed before approval")
        
        if self._is_rejected:
            raise ValueError("Cannot approve rejected video")
        
        self._is_approved = True
        self._is_flagged = False
        self._approved_at = datetime.utcnow()
        self._flag_reason = None
        self._flagged_at = None
        
        self._mark_updated()
        self._increment_version()
    
    def flag(self, reason: str) -> None:
        """Flag video for review."""
        if not reason:
            raise ValueError("Flag reason is required")
        
        self._is_flagged = True
        self._flag_reason = reason
        self._flagged_at = datetime.utcnow()
        
        self._mark_updated()
        self._increment_version()
    
    def reject(self, reason: str) -> None:
        """Reject video."""
        if not reason:
            raise ValueError("Rejection reason is required")
        
        self._is_rejected = True
        self._is_approved = False
        self._rejection_reason = reason
        self._rejected_at = datetime.utcnow()
        
        self._mark_updated()
        self._increment_version()
    
    def record_view(self, viewer_id: Optional[UUID], watch_time: float) -> None:
        """Record a video view."""
        if watch_time < 0:
            raise ValueError("Watch time cannot be negative")
        
        # Update engagement
        self._engagement = self._engagement.with_view()
        
        # Update watch time metrics
        self._total_watch_time += watch_time
        views = self._engagement.views
        if views > 0:
            self._average_watch_time = self._total_watch_time / views
            self._retention_rate = (self._average_watch_time / self._metadata.duration) * 100
        
        self._mark_updated()
        self._raise_event(VideoViewedEvent(self.id, viewer_id, watch_time))
    
    def record_like(self) -> None:
        """Record a like."""
        self._engagement = self._engagement.with_like()
        self._mark_updated()
    
    def record_comment(self) -> None:
        """Record a comment."""
        self._engagement = self._engagement.with_comment()
        self._mark_updated()
    
    def enable_monetization(self) -> None:
        """Enable monetization for this video."""
        if not self.is_ready():
            raise ValueError("Video must be approved and processed for monetization")
        
        self._is_monetized = True
        self._mark_updated()
        self._increment_version()
    
    def disable_monetization(self) -> None:
        """Disable monetization for this video."""
        self._is_monetized = False
        self._mark_updated()
        self._increment_version()
    
    def add_ad_revenue(self, amount: float) -> None:
        """Add ad revenue."""
        if amount < 0:
            raise ValueError("Revenue amount cannot be negative")
        
        if not self._is_monetized:
            raise ValueError("Video is not monetized")
        
        self._ad_revenue += amount
        self._mark_updated()
    
    def mark_deleted(self) -> None:
        """Mark video as deleted."""
        self._status = VideoStatus.DELETED
        self._visibility = VideoVisibility.PRIVATE
        self._mark_updated()
        self._increment_version()
