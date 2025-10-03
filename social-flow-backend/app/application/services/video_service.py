"""
Video Application Service

Orchestrates video-related use cases and workflows.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.video import VideoEntity
from app.domain.repositories.video_repository import IVideoRepository
from app.domain.value_objects import (
    VideoMetadata,
    StorageLocation,
)
from app.infrastructure.repositories import VideoRepository

logger = logging.getLogger(__name__)


class VideoApplicationService:
    """
    Video application service for video management use cases.
    
    Handles:
    - Video upload workflow
    - Processing coordination
    - Publishing and moderation
    - Discovery and recommendations
    - Monetization management
    - Engagement tracking
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
        self._video_repo: IVideoRepository = VideoRepository(session)
    
    # Video Upload & Processing
    
    async def initiate_video_upload(
        self,
        user_id: UUID,
        title: str,
        description: Optional[str],
        duration: int,
        file_size: int,
        format_type: str,
        resolution: str,
        storage_path: str,
        tags: Optional[List[str]] = None,
    ) -> VideoEntity:
        """
        Initiate video upload workflow.
        
        Args:
            user_id: Owner user ID
            title: Video title
            description: Video description
            duration: Duration in seconds
            file_size: File size in bytes
            format_type: Video format (mp4, webm, etc.)
            resolution: Video resolution (1080p, 720p, etc.)
            storage_path: Path where video is stored
            tags: Optional list of tags
            
        Returns:
            Created video entity in PROCESSING state
        """
        # Create metadata
        metadata = VideoMetadata(
            duration=duration,
            resolution=resolution,
            format=format_type,
            file_size=file_size,
        )
        
        # Create storage location
        storage = StorageLocation(
            provider="s3",  # Or your storage provider
            bucket="social-flow-videos",  # Your bucket name
            path=storage_path,
            region="us-west-2",  # Your region
        )
        
        # Create video entity
        video = VideoEntity(
            user_id=user_id,
            title=title,
            description=description,
            metadata=metadata,
            storage_location=storage,
        )
        
        # Add tags if provided
        if tags:
            for tag in tags:
                video.add_tag(tag)
        
        # Save
        saved_video = await self._video_repo.add(video)
        await self._session.commit()
        
        logger.info(f"Video upload initiated: {saved_video.id} by user {user_id}")
        
        return saved_video
    
    async def complete_video_processing(
        self,
        video_id: UUID,
        thumbnail_url: str,
        stream_url: str,
    ) -> VideoEntity:
        """
        Mark video processing as complete.
        
        Args:
            video_id: Video ID
            thumbnail_url: Generated thumbnail URL
            stream_url: Streaming URL
            
        Returns:
            Updated video entity in READY state
            
        Raises:
            ValueError: If video not found or invalid state
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        # Complete processing (domain validation)
        video.complete_processing(thumbnail_url, stream_url)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.info(f"Video processing complete: {video_id}")
        
        return updated_video
    
    async def fail_video_processing(
        self,
        video_id: UUID,
        error_message: str,
    ) -> VideoEntity:
        """
        Mark video processing as failed.
        
        Args:
            video_id: Video ID
            error_message: Error description
            
        Returns:
            Updated video entity in FAILED state
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.fail_processing(error_message)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.error(f"Video processing failed: {video_id} - {error_message}")
        
        return updated_video
    
    # Video Management
    
    async def get_video_by_id(self, video_id: UUID) -> Optional[VideoEntity]:
        """Get video by ID."""
        return await self._video_repo.get_by_id(video_id)
    
    async def get_user_videos(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get videos by user."""
        return await self._video_repo.get_by_owner(user_id, skip, limit)
    
    async def update_video_details(
        self,
        video_id: UUID,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> VideoEntity:
        """
        Update video details.
        
        Args:
            video_id: Video ID
            title: New title
            description: New description
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found or validation fails
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.update_details(title=title, description=description)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.info(f"Video details updated: {video_id}")
        
        return updated_video
    
    async def update_video_visibility(
        self,
        video_id: UUID,
        is_public: bool,
    ) -> VideoEntity:
        """
        Update video visibility.
        
        Args:
            video_id: Video ID
            is_public: Whether video should be public
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.set_visibility(is_public)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.info(f"Video visibility updated: {video_id} -> public={is_public}")
        
        return updated_video
    
    async def delete_video(self, video_id: UUID) -> bool:
        """
        Delete video.
        
        Args:
            video_id: Video ID
            
        Returns:
            True if deleted
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        await self._video_repo.delete(video_id)
        await self._session.commit()
        
        logger.info(f"Video deleted: {video_id}")
        
        return True
    
    # Engagement
    
    async def record_video_view(self, video_id: UUID) -> VideoEntity:
        """
        Record video view.
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.record_view()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        return updated_video
    
    async def like_video(self, video_id: UUID) -> VideoEntity:
        """
        Like video (increment likes).
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.increment_likes()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        return updated_video
    
    async def unlike_video(self, video_id: UUID) -> VideoEntity:
        """
        Unlike video (decrement likes).
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.decrement_likes()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        return updated_video
    
    async def add_video_comment(self, video_id: UUID) -> VideoEntity:
        """
        Add comment to video (increment comment count).
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.increment_comments()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        return updated_video
    
    async def share_video(self, video_id: UUID) -> VideoEntity:
        """
        Share video (increment shares).
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.increment_shares()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        return updated_video
    
    # Discovery
    
    async def discover_videos(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get videos for discovery feed."""
        return await self._video_repo.get_for_discovery(skip, limit)
    
    async def get_trending_videos(
        self,
        days: int = 7,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get trending videos."""
        return await self._video_repo.get_trending(days, skip, limit)
    
    async def get_top_videos(
        self,
        days: int = 30,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get top videos by views."""
        return await self._video_repo.get_top_by_views(days, skip, limit)
    
    async def search_videos(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Search videos by title."""
        return await self._video_repo.search_by_title(query, skip, limit)
    
    async def get_videos_by_tag(
        self,
        tag: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get videos by tag."""
        return await self._video_repo.get_by_tag(tag, skip, limit)
    
    # Moderation
    
    async def flag_video_for_review(
        self,
        video_id: UUID,
        reason: str,
    ) -> VideoEntity:
        """
        Flag video for moderation review.
        
        Args:
            video_id: Video ID
            reason: Flagging reason
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.flag_for_review(reason)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.warning(f"Video flagged: {video_id} - {reason}")
        
        return updated_video
    
    async def approve_video(self, video_id: UUID) -> VideoEntity:
        """
        Approve video after review.
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.approve()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.info(f"Video approved: {video_id}")
        
        return updated_video
    
    async def reject_video(
        self,
        video_id: UUID,
        reason: str,
    ) -> VideoEntity:
        """
        Reject video.
        
        Args:
            video_id: Video ID
            reason: Rejection reason
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.reject(reason)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.warning(f"Video rejected: {video_id} - {reason}")
        
        return updated_video
    
    async def get_flagged_videos(
        self,
        skip: int = 0,
        limit: int = 50,
    ) -> List[VideoEntity]:
        """Get videos flagged for review."""
        return await self._video_repo.get_flagged_for_review(skip, limit)
    
    # Monetization
    
    async def enable_video_monetization(
        self,
        video_id: UUID,
    ) -> VideoEntity:
        """
        Enable monetization for video.
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found or not eligible
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.enable_monetization()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.info(f"Monetization enabled: {video_id}")
        
        return updated_video
    
    async def disable_video_monetization(
        self,
        video_id: UUID,
    ) -> VideoEntity:
        """
        Disable monetization for video.
        
        Args:
            video_id: Video ID
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.disable_monetization()
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        logger.info(f"Monetization disabled: {video_id}")
        
        return updated_video
    
    async def record_video_revenue(
        self,
        video_id: UUID,
        amount: float,
    ) -> VideoEntity:
        """
        Record revenue for video.
        
        Args:
            video_id: Video ID
            amount: Revenue amount
            
        Returns:
            Updated video entity
            
        Raises:
            ValueError: If video not found or invalid amount
        """
        video = await self._video_repo.get_by_id(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found")
        
        video.record_revenue(amount)
        
        updated_video = await self._video_repo.update(video)
        await self._session.commit()
        
        return updated_video
    
    async def get_monetized_videos(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> List[VideoEntity]:
        """Get monetized videos by user."""
        return await self._video_repo.get_monetized_by_user(user_id, skip, limit)
    
    # Statistics
    
    async def get_video_count(self) -> int:
        """Get total video count."""
        return await self._video_repo.count()
    
    async def get_user_video_count(self, user_id: UUID) -> int:
        """Get video count for user."""
        return await self._video_repo.count_by_user(user_id)
    
    async def get_public_videos(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """Get public videos."""
        return await self._video_repo.get_public_videos(skip, limit)
