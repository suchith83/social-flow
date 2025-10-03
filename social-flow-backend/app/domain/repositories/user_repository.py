"""
User Repository Interface

Defines the contract for user data access operations.
"""

from abc import abstractmethod
from typing import List, Optional
from uuid import UUID

from app.domain.entities.user import UserEntity
from app.domain.repositories.base import IRepository
from app.domain.value_objects import Email, Username


class IUserRepository(IRepository[UserEntity]):
    """
    User repository interface.
    
    Defines operations specific to user entities.
    """
    
    @abstractmethod
    async def get_by_username(self, username: Username) -> Optional[UserEntity]:
        """
        Get user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_email(self, email: Email) -> Optional[UserEntity]:
        """
        Get user by email.
        
        Args:
            email: Email to search for
            
        Returns:
            User entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def username_exists(self, username: Username) -> bool:
        """
        Check if username exists.
        
        Args:
            username: Username to check
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def email_exists(self, email: Email) -> bool:
        """
        Check if email exists.
        
        Args:
            email: Email to check
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def search_by_username(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """
        Search users by username (partial match).
        
        Args:
            query: Search query
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of matching users
        """
        pass
    
    @abstractmethod
    async def get_followers(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """
        Get user's followers.
        
        Args:
            user_id: User ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of followers
        """
        pass
    
    @abstractmethod
    async def get_following(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """
        Get users that this user follows.
        
        Args:
            user_id: User ID
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of following users
        """
        pass
    
    @abstractmethod
    async def is_following(
        self,
        follower_id: UUID,
        following_id: UUID,
    ) -> bool:
        """
        Check if user is following another user.
        
        Args:
            follower_id: Follower user ID
            following_id: Following user ID
            
        Returns:
            True if following, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """
        Get active (not banned/suspended) users.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of active users
        """
        pass
    
    @abstractmethod
    async def get_verified_creators(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """
        Get verified creator users.
        
        Args:
            skip: Number of results to skip
            limit: Maximum number of results
            
        Returns:
            List of verified creators
        """
        pass
