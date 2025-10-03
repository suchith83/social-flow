"""
User Repository Implementation

SQLAlchemy-based implementation of IUserRepository interface.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User as UserModel
from app.models.social import Follow
from app.domain.entities.user import UserEntity
from app.domain.repositories.user_repository import IUserRepository
from app.domain.value_objects import Email, Username
from app.infrastructure.repositories.mappers import UserMapper


class UserRepository(IUserRepository):
    """
    SQLAlchemy implementation of user repository.
    
    Handles persistence and retrieval of user entities.
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
        self._mapper = UserMapper()
    
    async def get_by_id(self, id: UUID) -> Optional[UserEntity]:
        """Get user by ID."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._mapper.to_entity(model)
    
    async def get_by_username(self, username: Username) -> Optional[UserEntity]:
        """Get user by username."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.username == str(username))
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._mapper.to_entity(model)
    
    async def get_by_email(self, email: Email) -> Optional[UserEntity]:
        """Get user by email."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.email == str(email))
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._mapper.to_entity(model)
    
    async def username_exists(self, username: Username) -> bool:
        """Check if username exists."""
        result = await self._session.execute(
            select(func.count(UserModel.id))
            .where(UserModel.username == str(username))
        )
        count = result.scalar()
        return count > 0
    
    async def email_exists(self, email: Email) -> bool:
        """Check if email exists."""
        result = await self._session.execute(
            select(func.count(UserModel.id))
            .where(UserModel.email == str(email))
        )
        count = result.scalar()
        return count > 0
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """Get all users with pagination."""
        result = await self._session.execute(
            select(UserModel)
            .offset(skip)
            .limit(limit)
            .order_by(UserModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def search_by_username(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """Search users by username (partial match)."""
        result = await self._session.execute(
            select(UserModel)
            .where(UserModel.username.ilike(f"%{query}%"))
            .offset(skip)
            .limit(limit)
            .order_by(UserModel.username)
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_followers(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """Get user's followers."""
        result = await self._session.execute(
            select(UserModel)
            .join(Follow, Follow.follower_id == UserModel.id)
            .where(Follow.following_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(Follow.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_following(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """Get users that this user follows."""
        result = await self._session.execute(
            select(UserModel)
            .join(Follow, Follow.following_id == UserModel.id)
            .where(Follow.follower_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(Follow.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def is_following(
        self,
        follower_id: UUID,
        following_id: UUID,
    ) -> bool:
        """Check if user is following another user."""
        result = await self._session.execute(
            select(func.count(Follow.id))
            .where(
                Follow.follower_id == follower_id,
                Follow.following_id == following_id
            )
        )
        count = result.scalar()
        return count > 0
    
    async def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """Get active (not banned/suspended) users."""
        result = await self._session.execute(
            select(UserModel)
            .where(
                UserModel.is_active == True,
                UserModel.is_banned == False,
                UserModel.is_suspended == False
            )
            .offset(skip)
            .limit(limit)
            .order_by(UserModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_verified_creators(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """Get verified creator users."""
        result = await self._session.execute(
            select(UserModel)
            .where(
                UserModel.is_verified == True,
                UserModel.is_active == True
            )
            .offset(skip)
            .limit(limit)
            .order_by(UserModel.total_views.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def add(self, entity: UserEntity) -> UserEntity:
        """Add new user."""
        model = self._mapper.to_model(entity)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        
        return self._mapper.to_entity(model)
    
    async def update(self, entity: UserEntity) -> UserEntity:
        """Update existing user."""
        # Get existing model
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == entity.id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            raise ValueError(f"User with id {entity.id} not found")
        
        # Update model from entity
        model = self._mapper.to_model(entity, model)
        await self._session.flush()
        await self._session.refresh(model)
        
        return self._mapper.to_entity(model)
    
    async def delete(self, id: UUID) -> bool:
        """Delete user by ID."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self._session.delete(model)
        await self._session.flush()
        
        return True
    
    async def exists(self, id: UUID) -> bool:
        """Check if user exists."""
        result = await self._session.execute(
            select(func.count(UserModel.id)).where(UserModel.id == id)
        )
        count = result.scalar()
        return count > 0
    
    async def count(self) -> int:
        """Get total count of users."""
        result = await self._session.execute(
            select(func.count(UserModel.id))
        )
        return result.scalar()

