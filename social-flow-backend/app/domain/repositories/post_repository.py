"""
Post Repository Interface

Defines the contract for post data access operations.
"""

from abc import abstractmethod
from typing import List
from uuid import UUID

from app.domain.entities.post import PostEntity
from app.domain.repositories.base import IRepository
from app.domain.value_objects import PostVisibility


class IPostRepository(IRepository[PostEntity]):
    """
    Post repository interface.
    
    Defines operations specific to post entities.
    """
    
    @abstractmethod
    async def get_by_owner(
        self,
        owner_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get posts by owner.
        
        Args:
            owner_id: Owner user ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of posts
        """
        pass
    
    @abstractmethod
    async def get_public_posts(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get public posts.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of public posts
        """
        pass
    
    @abstractmethod
    async def get_feed_for_user(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get personalized feed for a user (posts from followed users).
        
        Args:
            user_id: User ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of posts
        """
        pass
    
    @abstractmethod
    async def get_trending_posts(
        self,
        hours: int = 24,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get trending posts based on recent engagement.
        
        Args:
            hours: Time window in hours
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of trending posts
        """
        pass
    
    @abstractmethod
    async def search_posts(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Search posts by content.
        
        Args:
            query: Search query
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of matching posts
        """
        pass
    
    @abstractmethod
    async def get_flagged_posts(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PostEntity]:
        """
        Get flagged posts for moderation.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of flagged posts
        """
        pass
    
    @abstractmethod
    async def get_by_visibility(
        self,
        visibility: PostVisibility,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PostEntity]:
        """
        Get posts by visibility level.
        
        Args:
            visibility: Post visibility
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of posts
        """
        pass
