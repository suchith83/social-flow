"""
Video Repository Interface

Defines the contract for video data access operations.
"""

from abc import abstractmethod
from typing import List, Optional
from uuid import UUID

from app.domain.entities.video import VideoEntity
from app.domain.repositories.base import IRepository
from app.domain.value_objects import VideoStatus, VideoVisibility


class IVideoRepository(IRepository[VideoEntity]):
    """
    Video repository interface.
    
    Defines operations specific to video entities.
    """
    
    @abstractmethod
    async def get_by_owner(
        self,
        owner_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Get videos by owner.
        
        Args:
            owner_id: Owner user ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of videos
        """
        pass
    
    @abstractmethod
    async def get_by_status(
        self,
        status: VideoStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """
        Get videos by status.
        
        Args:
            status: Video status
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of videos
        """
        pass
    
    @abstractmethod
    async def get_public_videos(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Get public videos.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of public videos
        """
        pass
    
    @abstractmethod
    async def get_trending_videos(
        self,
        hours: int = 24,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Get trending videos based on recent engagement.
        
        Args:
            hours: Time window in hours
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of trending videos
        """
        pass
    
    @abstractmethod
    async def get_recommended_for_user(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Get recommended videos for a user.
        
        Args:
            user_id: User ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of recommended videos
        """
        pass
    
    @abstractmethod
    async def search_by_title(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Search videos by title.
        
        Args:
            query: Search query
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of matching videos
        """
        pass
    
    @abstractmethod
    async def search_by_tags(
        self,
        tags: List[str],
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Search videos by tags.
        
        Args:
            tags: List of tags
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of matching videos
        """
        pass
    
    @abstractmethod
    async def get_flagged_videos(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """
        Get flagged videos for moderation.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of flagged videos
        """
        pass
    
    @abstractmethod
    async def get_pending_approval(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """
        Get videos pending approval.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of videos pending approval
        """
        pass
    
    @abstractmethod
    async def get_monetized_videos(
        self,
        owner_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """
        Get monetized videos for a user.
        
        Args:
            owner_id: Owner user ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of monetized videos
        """
        pass
    
    @abstractmethod
    async def get_total_views_for_user(self, owner_id: UUID) -> int:
        """
        Get total views across all user's videos.
        
        Args:
            owner_id: Owner user ID
            
        Returns:
            Total view count
        """
        pass
    
    @abstractmethod
    async def get_total_revenue_for_user(self, owner_id: UUID) -> float:
        """
        Get total ad revenue across all user's videos.
        
        Args:
            owner_id: Owner user ID
            
        Returns:
            Total revenue
        """
        pass
