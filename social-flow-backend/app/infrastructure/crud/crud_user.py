"""
CRUD operations for User model.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash, verify_password
from app.infrastructure.crud.base import CRUDBase
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """CRUD operations for User model."""

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: UserCreate,
        commit: bool = True,
    ) -> User:
        """
        Create a new user with password hashing.
        
        Args:
            db: Database session
            obj_in: Pydantic schema with user data (including plaintext password)
            commit: Whether to commit the transaction
            
        Returns:
            Created user instance
        """
        # Convert schema to dict and hash password
        user_data = obj_in.model_dump(exclude={"password", "full_name", "website_url"})
        user_data["password_hash"] = get_password_hash(obj_in.password)
        
        # Handle display_name/full_name mapping (prefer display_name)
        if not user_data.get("display_name") and obj_in.full_name:
            user_data["display_name"] = obj_in.full_name
        
        # Map website_url to website
        if hasattr(obj_in, 'website_url') and obj_in.website_url:
            user_data["website"] = obj_in.website_url
        
        # Create user model instance
        db_obj = self.model(**user_data)
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_obj

    async def get_by_email(
        self,
        db: AsyncSession,
        *,
        email: str,
    ) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            db: Database session
            email: User email address
            
        Returns:
            User instance or None if not found
        """
        return await self.get_by_field(db, "email", email)

    async def get_by_username(
        self,
        db: AsyncSession,
        *,
        username: str,
    ) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            db: Database session
            username: Username
            
        Returns:
            User instance or None if not found
        """
        return await self.get_by_field(db, "username", username)

    async def get_by_email_or_username(
        self,
        db: AsyncSession,
        *,
        identifier: str,
    ) -> Optional[User]:
        """
        Get user by email or username.
        
        Args:
            db: Database session
            identifier: Email or username
            
        Returns:
            User instance or None if not found
        """
        query = select(self.model).where(
            or_(
                self.model.email == identifier,
                self.model.username == identifier,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def authenticate(
        self,
        db: AsyncSession,
        *,
        email: str,
        password: str,
    ) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Args:
            db: Database session
            email: User email
            password: Plain text password
            
        Returns:
            User instance if authenticated, None otherwise
        """
        user = await self.get_by_email(db, email=email)
        if not user:
            return None
        if not user.verify_password(password):
            return None
        return user

    async def is_email_taken(
        self,
        db: AsyncSession,
        *,
        email: str,
        exclude_id: Optional[UUID] = None,
    ) -> bool:
        """
        Check if email is already taken.
        
        Args:
            db: Database session
            email: Email to check
            exclude_id: User ID to exclude from check (for updates)
            
        Returns:
            True if email is taken, False otherwise
        """
        query = select(self.model).where(self.model.email == email)
        if exclude_id:
            query = query.where(self.model.id != exclude_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none() is not None

    async def is_username_taken(
        self,
        db: AsyncSession,
        *,
        username: str,
        exclude_id: Optional[UUID] = None,
    ) -> bool:
        """
        Check if username is already taken.
        
        Args:
            db: Database session
            username: Username to check
            exclude_id: User ID to exclude from check (for updates)
            
        Returns:
            True if username is taken, False otherwise
        """
        query = select(self.model).where(self.model.username == username)
        if exclude_id:
            query = query.where(self.model.id != exclude_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none() is not None

    async def update_last_login(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> Optional[User]:
        """
        Update user's last login timestamp.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None if not found
        """
        from datetime import datetime, timezone
        
        user = await self.get(db, user_id)
        if not user:
            return None
        
        user.last_login_at = datetime.now(timezone.utc)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

    async def activate_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> Optional[User]:
        """
        Activate a user account.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None if not found
        """
        from app.models.user import UserStatus
        
        user = await self.get(db, user_id)
        if not user:
            return None
        
        user.status = UserStatus.ACTIVE
        user.email_verified = True
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

    async def deactivate_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> Optional[User]:
        """
        Deactivate a user account.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None if not found
        """
        from app.models.user import UserStatus
        
        user = await self.get(db, user_id)
        if not user:
            return None
        
        user.status = UserStatus.INACTIVE
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

    async def suspend_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> Optional[User]:
        """
        Suspend a user account.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None if not found
        """
        from app.models.user import UserStatus
        
        user = await self.get(db, user_id)
        if not user:
            return None
        
        user.status = UserStatus.SUSPENDED
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

    async def get_followers(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ):
        """
        Get user's followers.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of follower users
        """
        from app.models.social import Follow
        
        query = (
            select(self.model)
            .join(Follow, Follow.follower_id == self.model.id)
            .where(Follow.following_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_following(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ):
        """
        Get users that this user is following.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of followed users
        """
        from app.models.social import Follow
        
        query = (
            select(self.model)
            .join(Follow, Follow.following_id == self.model.id)
            .where(Follow.follower_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_followers_count(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> int:
        """
        Get count of user's followers.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Number of followers
        """
        from sqlalchemy import func
        from app.models.social import Follow
        
        query = select(func.count()).select_from(Follow).where(Follow.following_id == user_id)
        result = await db.execute(query)
        return result.scalar_one()

    async def get_following_count(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> int:
        """
        Get count of users this user is following.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Number of users being followed
        """
        from sqlalchemy import func
        from app.models.social import Follow
        
        query = select(func.count()).select_from(Follow).where(Follow.follower_id == user_id)
        result = await db.execute(query)
        return result.scalar_one()


# Create singleton instance
user = CRUDUser(User)
