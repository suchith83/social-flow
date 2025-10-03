"""
User Repository Interface - Auth Bounded Context

Defines the contract for user persistence operations.
This is a Protocol (interface) that infrastructure implementations must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from uuid import UUID

from app.auth.domain.entities import UserEntity
from app.auth.domain.value_objects import Email, Username


class IUserRepository(ABC):
    """
    Repository interface for User aggregate.
    
    Defines the contract for user persistence operations.
    Infrastructure layer will provide the concrete implementation.
    """
    
    @abstractmethod
    async def save(self, user: UserEntity) -> None:
        """
        Persist a user entity.
        
        Args:
            user: The user entity to save
            
        Raises:
            ValueError: If user already exists (on create)
            RepositoryError: If save operation fails
        """
        pass
    
    @abstractmethod
    async def update(self, user: UserEntity) -> None:
        """
        Update an existing user entity.
        
        Args:
            user: The user entity to update
            
        Raises:
            ValueError: If user doesn't exist
            RepositoryError: If update operation fails
        """
        pass
    
    @abstractmethod
    async def delete(self, user_id: UUID) -> None:
        """
        Delete a user by ID.
        
        Args:
            user_id: The ID of the user to delete
            
        Raises:
            ValueError: If user doesn't exist
            RepositoryError: If delete operation fails
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, user_id: UUID) -> Optional[UserEntity]:
        """
        Find a user by ID.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            UserEntity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[UserEntity]:
        """
        Find a user by email address.
        
        Args:
            email: The user's email (value object)
            
        Returns:
            UserEntity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_username(self, username: Username) -> Optional[UserEntity]:
        """
        Find a user by username.
        
        Args:
            username: The user's username (value object)
            
        Returns:
            UserEntity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def exists_by_email(self, email: Email) -> bool:
        """
        Check if a user exists with the given email.
        
        Args:
            email: The email to check
            
        Returns:
            True if user exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists_by_username(self, username: Username) -> bool:
        """
        Check if a user exists with the given username.
        
        Args:
            username: The username to check
            
        Returns:
            True if user exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """
        Find all users with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of UserEntity instances
        """
        pass
    
    @abstractmethod
    async def find_by_role(
        self,
        role: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """
        Find users by role with pagination.
        
        Args:
            role: The role to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of UserEntity instances with the specified role
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Count total number of users.
        
        Returns:
            Total user count
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: str) -> int:
        """
        Count users by account status.
        
        Args:
            status: The account status to count
            
        Returns:
            Number of users with the specified status
        """
        pass
