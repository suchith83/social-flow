"""
Authentication service.

This module contains authentication business logic.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, ValidationError
from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.auth import UserCreate, UserUpdate


class AuthService:
    """Authentication service for user management."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            display_name=user_data.display_name,
            bio=user_data.bio,
            avatar_url=user_data.avatar_url,
            website=user_data.website,
            location=user_data.location,
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Update fields
        for field, value in user_data.dict(exclude_unset=True).items():
            if field == "password":
                user.password_hash = get_password_hash(value)
            else:
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def update_password(self, user_id: str, new_password: str) -> bool:
        """Update user password."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.password_hash = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.last_login_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        # Try to find user by username or email
        user = await self.get_user_by_username(username)
        if not user:
            user = await self.get_user_by_email(username)
        
        if not user:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        return user
    
    async def verify_user(self, user_id: str) -> bool:
        """Verify user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_verified = True
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def ban_user(self, user_id: str, reason: str) -> bool:
        """Ban user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_banned = True
        user.ban_reason = reason
        user.banned_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def unban_user(self, user_id: str) -> bool:
        """Unban user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_banned = False
        user.ban_reason = None
        user.banned_at = None
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def suspend_user(self, user_id: str, reason: str, duration_days: int = None) -> bool:
        """Suspend user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_suspended = True
        user.suspension_reason = reason
        user.suspended_at = datetime.utcnow()
        
        if duration_days:
            from datetime import timedelta
            user.suspension_ends_at = datetime.utcnow() + timedelta(days=duration_days)
        
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def unsuspend_user(self, user_id: str) -> bool:
        """Unsuspend user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_suspended = False
        user.suspension_reason = None
        user.suspended_at = None
        user.suspension_ends_at = None
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        await self.db.delete(user)
        await self.db.commit()
        return True
