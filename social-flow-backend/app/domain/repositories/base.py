"""
Base Repository Interface

Defines common repository operations for all entities.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from uuid import UUID


# Generic type for entities
EntityType = TypeVar("EntityType")


class IRepository(ABC, Generic[EntityType]):
    """
    Base repository interface.
    
    Provides common CRUD operations for all repositories.
    All repositories should inherit from this interface.
    """
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[EntityType]:
        """
        Get entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[EntityType]:
        """
        Get all entities with pagination.
        
        Args:
            skip: Number of entities to skip
            limit: Maximum number of entities to return
            
        Returns:
            List of entities
        """
        pass
    
    @abstractmethod
    async def add(self, entity: EntityType) -> EntityType:
        """
        Add new entity.
        
        Args:
            entity: Entity to add
            
        Returns:
            Added entity
        """
        pass
    
    @abstractmethod
    async def update(self, entity: EntityType) -> EntityType:
        """
        Update existing entity.
        
        Args:
            entity: Entity to update
            
        Returns:
            Updated entity
        """
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """
        Delete entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            True if deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists(self, id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            id: Entity ID
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Get total count of entities.
        
        Returns:
            Total count
        """
        pass
