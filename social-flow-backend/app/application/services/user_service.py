"""
User Application Service

Orchestrates user-related use cases and business workflows.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash, verify_password
from app.domain.entities.user import UserEntity
from app.domain.repositories.user_repository import IUserRepository
from app.domain.value_objects import Email, Username, UserRole
from app.infrastructure.repositories import UserRepository

logger = logging.getLogger(__name__)


class UserApplicationService:
    """
    User application service for user management use cases.
    
    Handles:
    - User registration
    - Authentication
    - Profile management
    - Social features (follow/unfollow)
    - Account management
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
        self._user_repo: IUserRepository = UserRepository(session)
    
    # Registration & Authentication
    
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        display_name: str,
    ) -> UserEntity:
        """
        Register a new user.
        
        Args:
            username: Desired username
            email: Email address
            password: Plain text password (will be hashed)
            display_name: Display name
            
        Returns:
            Created user entity
            
        Raises:
            ValueError: If username or email already exists or validation fails
        """
        try:
            # Validate and create value objects
            username_vo = Username(username)
            email_vo = Email(email)
            
        except ValueError as e:
            logger.warning(f"Validation failed during registration: {e}")
            raise ValueError(f"Invalid input: {e}")
        
        # Check uniqueness
        if await self._user_repo.username_exists(username_vo):
            raise ValueError(f"Username '{username}' already exists")
        
        if await self._user_repo.email_exists(email_vo):
            raise ValueError(f"Email '{email}' already registered")
        
        # Hash password
        password_hash = get_password_hash(password)
        
        # Create entity
        user = UserEntity(
            username=username_vo,
            email=email_vo,
            password_hash=password_hash,
            display_name=display_name,
        )
        
        # Save
        saved_user = await self._user_repo.add(user)
        await self._session.commit()
        
        logger.info(f"User registered: {saved_user.id} ({username})")
        
        return saved_user
    
    async def authenticate_user(
        self,
        username_or_email: str,
        password: str,
    ) -> Optional[UserEntity]:
        """
        Authenticate user with username/email and password.
        
        Args:
            username_or_email: Username or email
            password: Plain text password
            
        Returns:
            User entity if authenticated, None otherwise
        """
        # Try to find user by username or email
        user = None
        
        try:
            if "@" in username_or_email:
                email_vo = Email(username_or_email)
                user = await self._user_repo.get_by_email(email_vo)
            else:
                username_vo = Username(username_or_email)
                user = await self._user_repo.get_by_username(username_vo)
        except ValueError:
            # Invalid format
            return None
        
        if user is None:
            return None
        
        # Verify password
        if not verify_password(password, user.password_hash):
            return None
        
        # Check account status
        if not user.is_account_active():
            logger.warning(f"Login attempt for inactive account: {user.id}")
            return None
        
        # Record login
        user.record_login()
        await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"User authenticated: {user.id}")
        
        return user
    
    # Profile Management
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[UserEntity]:
        """Get user by ID."""
        return await self._user_repo.get_by_id(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[UserEntity]:
        """Get user by username."""
        try:
            username_vo = Username(username)
            return await self._user_repo.get_by_username(username_vo)
        except ValueError:
            return None
    
    async def update_user_profile(
        self,
        user_id: UUID,
        display_name: Optional[str] = None,
        bio: Optional[str] = None,
        website: Optional[str] = None,
        location: Optional[str] = None,
    ) -> UserEntity:
        """
        Update user profile.
        
        Args:
            user_id: User ID
            display_name: New display name
            bio: New bio
            website: New website URL
            location: New location
            
        Returns:
            Updated user entity
            
        Raises:
            ValueError: If user not found or validation fails
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        # Update profile (domain validation applied)
        user.update_profile(
            display_name=display_name,
            bio=bio,
            website=website,
            location=location,
        )
        
        # Save
        updated_user = await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"Profile updated: {user_id}")
        
        return updated_user
    
    async def update_user_avatar(
        self,
        user_id: UUID,
        avatar_url: str,
    ) -> UserEntity:
        """
        Update user avatar.
        
        Args:
            user_id: User ID
            avatar_url: New avatar URL
            
        Returns:
            Updated user entity
            
        Raises:
            ValueError: If user not found
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        user.update_avatar(avatar_url)
        
        updated_user = await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"Avatar updated: {user_id}")
        
        return updated_user
    
    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password
            
        Returns:
            True if changed successfully
            
        Raises:
            ValueError: If user not found or current password incorrect
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        # Verify current password
        if not verify_password(current_password, user.password_hash):
            raise ValueError("Current password is incorrect")
        
        # Hash new password
        new_password_hash = get_password_hash(new_password)
        
        # Change password (domain validation)
        user.change_password(new_password_hash)
        
        await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"Password changed: {user_id}")
        
        return True
    
    # Social Features
    
    async def search_users(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """
        Search users by username.
        
        Args:
            query: Search query
            skip: Number of results to skip
            limit: Maximum results to return
            
        Returns:
            List of matching users
        """
        return await self._user_repo.search_by_username(query, skip, limit)
    
    async def get_followers(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """Get user's followers."""
        return await self._user_repo.get_followers(user_id, skip, limit)
    
    async def get_following(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[UserEntity]:
        """Get users that this user follows."""
        return await self._user_repo.get_following(user_id, skip, limit)
    
    async def is_following(
        self,
        follower_id: UUID,
        following_id: UUID,
    ) -> bool:
        """Check if user is following another user."""
        return await self._user_repo.is_following(follower_id, following_id)
    
    # Account Management
    
    async def verify_user_account(self, user_id: UUID) -> UserEntity:
        """
        Verify user account.
        
        Args:
            user_id: User ID
            
        Returns:
            Updated user entity
            
        Raises:
            ValueError: If user not found
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        user.verify_account()
        
        updated_user = await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"Account verified: {user_id}")
        
        return updated_user
    
    async def promote_user_role(
        self,
        user_id: UUID,
        new_role: UserRole,
    ) -> UserEntity:
        """
        Promote user to a new role.
        
        Args:
            user_id: User ID
            new_role: New role
            
        Returns:
            Updated user entity
            
        Raises:
            ValueError: If user not found or invalid role transition
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        user.promote_role(new_role)
        
        updated_user = await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"User role promoted: {user_id} -> {new_role}")
        
        return updated_user
    
    async def ban_user(
        self,
        user_id: UUID,
        reason: str,
    ) -> UserEntity:
        """
        Ban user account.
        
        Args:
            user_id: User ID
            reason: Ban reason
            
        Returns:
            Updated user entity
            
        Raises:
            ValueError: If user not found
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        user.ban(reason)
        
        updated_user = await self._user_repo.update(user)
        await self._session.commit()
        
        logger.warning(f"User banned: {user_id} - {reason}")
        
        return updated_user
    
    async def unban_user(self, user_id: UUID) -> UserEntity:
        """
        Unban user account.
        
        Args:
            user_id: User ID
            
        Returns:
            Updated user entity
            
        Raises:
            ValueError: If user not found
        """
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")
        
        user.unban()
        
        updated_user = await self._user_repo.update(user)
        await self._session.commit()
        
        logger.info(f"User unbanned: {user_id}")
        
        return updated_user
    
    # Statistics
    
    async def get_user_count(self) -> int:
        """Get total user count."""
        return await self._user_repo.count()
    
    async def get_active_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """Get active users."""
        return await self._user_repo.get_active_users(skip, limit)
    
    async def get_verified_creators(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[UserEntity]:
        """Get verified creator users."""
        return await self._user_repo.get_verified_creators(skip, limit)
