"""
Post Domain Entity

Rich domain model for Post with business logic and invariants.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from app.domain.entities.base import AggregateRoot, DomainEvent
from app.domain.value_objects import EngagementMetrics, PostVisibility


class PostCreatedEvent(DomainEvent):
    """Event raised when a post is created."""
    
    def __init__(self, post_id: UUID, owner_id: UUID, content_preview: str):
        super().__init__(
            event_type="post.created",
            data={
                "post_id": str(post_id),
                "owner_id": str(owner_id),
                "content_preview": content_preview[:100],
            }
        )


class PostPublishedEvent(DomainEvent):
    """Event raised when a post is published."""
    
    def __init__(self, post_id: UUID, owner_id: UUID):
        super().__init__(
            event_type="post.published",
            data={
                "post_id": str(post_id),
                "owner_id": str(owner_id),
            }
        )


class PostEntity(AggregateRoot):
    """
    Post domain entity with business logic.
    
    Encapsulates post-related business rules and invariants.
    """
    
    def __init__(
        self,
        content: str,
        owner_id: UUID,
        id: Optional[UUID] = None,
    ):
        super().__init__(id)
        
        # Validate content
        if not content or len(content) < 1:
            raise ValueError("Post content cannot be empty")
        if len(content) > 10000:
            raise ValueError("Post content must be max 10000 characters")
        
        self._content = content
        self._owner_id = owner_id
        
        # Attachments
        self._image_urls: List[str] = []
        self._video_ids: List[UUID] = []
        
        # Status
        self._visibility = PostVisibility.PUBLIC
        self._is_published = False
        self._published_at: Optional[datetime] = None
        
        # Moderation
        self._is_flagged = False
        self._is_removed = False
        self._flag_reason: Optional[str] = None
        self._flagged_at: Optional[datetime] = None
        self._removed_at: Optional[datetime] = None
        self._removal_reason: Optional[str] = None
        
        # Engagement
        self._engagement = EngagementMetrics()
        
        # Raise creation event
        self._raise_event(PostCreatedEvent(self.id, owner_id, content))
    
    # Properties (getters)
    
    @property
    def content(self) -> str:
        return self._content
    
    @property
    def owner_id(self) -> UUID:
        return self._owner_id
    
    @property
    def image_urls(self) -> List[str]:
        return self._image_urls.copy()
    
    @property
    def video_ids(self) -> List[UUID]:
        return self._video_ids.copy()
    
    @property
    def visibility(self) -> PostVisibility:
        return self._visibility
    
    @property
    def is_published(self) -> bool:
        return self._is_published
    
    @property
    def published_at(self) -> Optional[datetime]:
        return self._published_at
    
    @property
    def is_flagged(self) -> bool:
        return self._is_flagged
    
    @property
    def is_removed(self) -> bool:
        return self._is_removed
    
    @property
    def engagement(self) -> EngagementMetrics:
        return self._engagement
    
    # Business logic methods
    
    def is_visible_to_public(self) -> bool:
        """Check if post is visible to public."""
        return (
            self._is_published
            and self._visibility == PostVisibility.PUBLIC
            and not self._is_removed
        )
    
    def can_be_viewed_by(self, user_id: Optional[UUID], is_friend: bool = False) -> bool:
        """Check if post can be viewed by a user."""
        # Removed posts are not visible
        if self._is_removed:
            return False
        
        # Owner can always view
        if user_id == self._owner_id:
            return True
        
        # Must be published
        if not self._is_published:
            return False
        
        # Check visibility
        if self._visibility == PostVisibility.PUBLIC:
            return True
        elif self._visibility == PostVisibility.FRIENDS:
            return is_friend
        else:  # PRIVATE
            return False
    
    def can_be_edited_by(self, user_id: UUID) -> bool:
        """Check if post can be edited by a user."""
        return user_id == self._owner_id and not self._is_removed
    
    def can_be_deleted_by(self, user_id: UUID) -> bool:
        """Check if post can be deleted by a user."""
        return user_id == self._owner_id
    
    def has_attachments(self) -> bool:
        """Check if post has any attachments."""
        return len(self._image_urls) > 0 or len(self._video_ids) > 0
    
    # Mutation methods
    
    def update_content(self, new_content: str) -> None:
        """Update post content."""
        if not new_content or len(new_content) < 1:
            raise ValueError("Post content cannot be empty")
        if len(new_content) > 10000:
            raise ValueError("Post content must be max 10000 characters")
        
        if self._is_removed:
            raise ValueError("Cannot update removed post")
        
        self._content = new_content
        self._mark_updated()
        self._increment_version()
    
    def add_images(self, image_urls: List[str]) -> None:
        """Add images to post."""
        if self._is_removed:
            raise ValueError("Cannot add images to removed post")
        
        if len(self._image_urls) + len(image_urls) > 10:
            raise ValueError("Maximum 10 images per post")
        
        self._image_urls.extend(image_urls)
        self._mark_updated()
        self._increment_version()
    
    def remove_image(self, image_url: str) -> None:
        """Remove an image from post."""
        if image_url in self._image_urls:
            self._image_urls.remove(image_url)
            self._mark_updated()
            self._increment_version()
    
    def attach_video(self, video_id: UUID) -> None:
        """Attach a video to post."""
        if self._is_removed:
            raise ValueError("Cannot attach video to removed post")
        
        if len(self._video_ids) >= 1:
            raise ValueError("Maximum 1 video per post")
        
        self._video_ids.append(video_id)
        self._mark_updated()
        self._increment_version()
    
    def set_visibility(self, visibility: PostVisibility) -> None:
        """Set post visibility."""
        self._visibility = visibility
        self._mark_updated()
        self._increment_version()
    
    def publish(self) -> None:
        """Publish the post."""
        if self._is_published:
            return
        
        if self._is_removed:
            raise ValueError("Cannot publish removed post")
        
        self._is_published = True
        self._published_at = datetime.utcnow()
        self._mark_updated()
        self._increment_version()
        self._raise_event(PostPublishedEvent(self.id, self._owner_id))
    
    def unpublish(self) -> None:
        """Unpublish the post."""
        if not self._is_published:
            return
        
        self._is_published = False
        self._mark_updated()
        self._increment_version()
    
    def flag(self, reason: str) -> None:
        """Flag post for review."""
        if not reason:
            raise ValueError("Flag reason is required")
        
        self._is_flagged = True
        self._flag_reason = reason
        self._flagged_at = datetime.utcnow()
        self._mark_updated()
        self._increment_version()
    
    def unflag(self) -> None:
        """Remove flag from post."""
        self._is_flagged = False
        self._flag_reason = None
        self._flagged_at = None
        self._mark_updated()
        self._increment_version()
    
    def remove(self, reason: str) -> None:
        """Remove post."""
        if not reason:
            raise ValueError("Removal reason is required")
        
        self._is_removed = True
        self._is_published = False
        self._removal_reason = reason
        self._removed_at = datetime.utcnow()
        self._mark_updated()
        self._increment_version()
    
    def record_view(self) -> None:
        """Record a view."""
        self._engagement = self._engagement.with_view()
        self._mark_updated()
    
    def record_like(self) -> None:
        """Record a like."""
        self._engagement = self._engagement.with_like()
        self._mark_updated()
    
    def record_comment(self) -> None:
        """Record a comment."""
        self._engagement = self._engagement.with_comment()
        self._mark_updated()
