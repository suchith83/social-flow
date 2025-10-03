"""
SQLAlchemy User Repository - Auth Infrastructure

Implements IUserRepository using SQLAlchemy for persistence.
Handles entity-to-model mapping and database operations.
"""

from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.auth.domain.repositories import IUserRepository
from app.auth.domain.entities import UserEntity
from app.auth.domain.value_objects import Email, Username
from app.auth.infrastructure.persistence.models import UserModel
from app.auth.infrastructure.persistence.mapper import UserMapper


class SQLAlchemyUserRepository(IUserRepository):
    """
    SQLAlchemy implementation of IUserRepository.
    
    Handles persistence of User aggregates using SQLAlchemy ORM.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self._session = session
        self._mapper = UserMapper()
    
    async def save(self, user: UserEntity) -> None:
        """
        Save a new user entity.
        
        Args:
            user: The user entity to save
            
        Raises:
            ValueError: If user already exists
        """
        try:
            # Convert entity to model
            model = self._mapper.to_model(user)
            
            # Add to session
            self._session.add(model)
            
            # Flush to get any database-generated values
            await self._session.flush()
            
        except IntegrityError as e:
            await self._session.rollback()
            if "username" in str(e.orig):
                raise ValueError(f"Username '{user.username.value}' already exists")
            elif "email" in str(e.orig):
                raise ValueError(f"Email '{user.email.value}' already exists")
            else:
                raise ValueError("User already exists") from e
    
    async def update(self, user: UserEntity) -> None:
        """
        Update an existing user entity.
        
        Args:
            user: The user entity to update
            
        Raises:
            ValueError: If user doesn't exist
        """
        # Find existing model
        stmt = select(UserModel).where(UserModel.id == user.id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            raise ValueError(f"User with ID {user.id} not found")
        
        # Check version for optimistic locking
        if model.version != user.version - 1:
            raise ValueError(
                f"Version mismatch: expected {model.version}, "
                f"got {user.version - 1}. User may have been modified."
            )
        
        # Update model from entity
        self._mapper.update_model_from_entity(user, model)
        
        # Flush changes
        await self._session.flush()
    
    async def delete(self, user_id: UUID) -> None:
        """
        Delete a user by ID.
        
        Args:
            user_id: The ID of the user to delete
            
        Raises:
            ValueError: If user doesn't exist
        """
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            raise ValueError(f"User with ID {user_id} not found")
        
        await self._session.delete(model)
        await self._session.flush()
    
    async def find_by_id(self, user_id: UUID) -> Optional[UserEntity]:
        """
        Find a user by ID.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            UserEntity if found, None otherwise
        """
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._mapper.to_domain(model)
    
    async def find_by_email(self, email: Email) -> Optional[UserEntity]:
        """
        Find a user by email address.
        
        Args:
            email: The user's email (value object)
            
        Returns:
            UserEntity if found, None otherwise
        """
        stmt = select(UserModel).where(UserModel.email == email.value)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._mapper.to_domain(model)
    
    async def find_by_username(self, username: Username) -> Optional[UserEntity]:
        """
        Find a user by username.
        
        Args:
            username: The user's username (value object)
            
        Returns:
            UserEntity if found, None otherwise
        """
        stmt = select(UserModel).where(UserModel.username == username.value)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._mapper.to_domain(model)
    
    async def exists_by_email(self, email: Email) -> bool:
        """
        Check if a user exists with the given email.
        
        Args:
            email: The email to check
            
        Returns:
            True if user exists, False otherwise
        """
        stmt = select(func.count()).select_from(UserModel).where(
            UserModel.email == email.value
        )
        result = await self._session.execute(stmt)
        count = result.scalar()
        return count > 0
    
    async def exists_by_username(self, username: Username) -> bool:
        """
        Check if a user exists with the given username.
        
        Args:
            username: The username to check
            
        Returns:
            True if user exists, False otherwise
        """
        stmt = select(func.count()).select_from(UserModel).where(
            UserModel.username == username.value
        )
        result = await self._session.execute(stmt)
        count = result.scalar()
        return count > 0
    
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
        stmt = select(UserModel).offset(skip).limit(limit)
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        
        return [self._mapper.to_domain(model) for model in models]
    
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
        stmt = (
            select(UserModel)
            .where(UserModel.role == role)
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        
        return [self._mapper.to_domain(model) for model in models]
    
    async def count(self) -> int:
        """
        Count total number of users.
        
        Returns:
            Total user count
        """
        stmt = select(func.count()).select_from(UserModel)
        result = await self._session.execute(stmt)
        return result.scalar() or 0
    
    async def count_by_status(self, status: str) -> int:
        """
        Count users by account status.
        
        Args:
            status: The account status to count
            
        Returns:
            Number of users with the specified status
        """
        stmt = (
            select(func.count())
            .select_from(UserModel)
            .where(UserModel.account_status == status)
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0
