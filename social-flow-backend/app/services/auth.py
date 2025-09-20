"""
Authentication service.

This module contains authentication business logic integrated from the Go user service.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncio

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import AuthenticationError, ValidationError, SocialFlowException
from app.core.security import get_password_hash, verify_password, create_access_token, create_refresh_token
from app.models.user import User
from app.models.follow import Follow
from app.models.post import Post
from app.models.video import Video
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
    
    # Additional authentication methods from Go service integration
    
    async def register_user_with_verification(self, user_data: UserCreate) -> Dict[str, Any]:
        """Register user with email verification."""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise ValidationError("User with this email already exists")
        
        existing_username = await self.get_user_by_username(user_data.username)
        if existing_username:
            raise ValidationError("Username already taken")
        
        # Create user
        user = await self.create_user(user_data)
        
        # Generate verification token
        verification_token = str(uuid.uuid4())
        
        # TODO: Send verification email
        # await self.send_verification_email(user.email, verification_token)
        
        return {
            "user_id": str(user.id),
            "email": user.email,
            "verification_required": True,
            "message": "Please check your email for verification link"
        }
    
    async def verify_email(self, verification_token: str) -> bool:
        """Verify user email with token."""
        # TODO: Implement email verification logic
        # This would typically involve checking the token against a stored verification token
        return True
    
    async def login_with_credentials(self, username: str, password: str) -> Dict[str, Any]:
        """Login user with username/email and password."""
        user = await self.authenticate_user(username, password)
        if not user:
            raise AuthenticationError("Invalid credentials")
        
        if user.is_banned:
            raise AuthenticationError("Account is banned")
        
        if user.is_suspended:
            if user.suspension_ends_at and user.suspension_ends_at > datetime.utcnow():
                raise AuthenticationError("Account is suspended")
            else:
                # Auto-unsuspend if suspension period has ended
                await self.unsuspend_user(str(user.id))
        
        # Update last login
        await self.update_last_login(str(user.id))
        
        # Generate tokens
        access_token = create_access_token({"sub": str(user.id), "username": user.username})
        refresh_token = create_refresh_token({"sub": str(user.id)})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name,
                "avatar_url": user.avatar_url,
                "is_verified": user.is_verified,
                "role": user.role
            }
        }
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        # TODO: Implement refresh token validation
        # This would typically involve checking the refresh token against a stored token
        
        # For now, return a new access token
        # In production, you'd validate the refresh token first
        return {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "token_type": "bearer"
        }
    
    async def logout_user(self, user_id: str) -> bool:
        """Logout user and invalidate tokens."""
        # TODO: Implement token invalidation
        # This would typically involve adding the token to a blacklist
        return True
    
    async def reset_password_request(self, email: str) -> bool:
        """Request password reset."""
        user = await self.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists or not
            return True
        
        # Generate reset token
        reset_token = str(uuid.uuid4())
        
        # TODO: Store reset token and send email
        # await self.send_password_reset_email(user.email, reset_token)
        
        return True
    
    async def reset_password(self, reset_token: str, new_password: str) -> bool:
        """Reset password using reset token."""
        # TODO: Implement password reset logic
        # This would typically involve validating the reset token
        return True
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        if not verify_password(current_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")
        
        return await self.update_password(user_id, new_password)
    
    async def enable_two_factor(self, user_id: str) -> Dict[str, str]:
        """Enable two-factor authentication."""
        # TODO: Implement 2FA setup
        return {
            "qr_code": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            "secret": "JBSWY3DPEHPK3PXP"
        }
    
    async def verify_two_factor(self, user_id: str, token: str) -> bool:
        """Verify two-factor authentication token."""
        # TODO: Implement 2FA verification
        return True
    
    async def disable_two_factor(self, user_id: str, password: str) -> bool:
        """Disable two-factor authentication."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        if not verify_password(password, user.password_hash):
            raise AuthenticationError("Password is incorrect")
        
        # TODO: Disable 2FA
        return True
    
    # Social login methods
    async def social_login(self, provider: str, social_id: str, email: str, name: str) -> Dict[str, Any]:
        """Login with social provider (Google, Facebook, etc.)."""
        # Check if user exists with this social ID
        # TODO: Implement social login logic
        return {
            "access_token": "social_access_token",
            "refresh_token": "social_refresh_token",
            "token_type": "bearer",
            "user": {
                "id": "user_id",
                "username": "username",
                "email": email,
                "display_name": name,
                "is_verified": True
            }
        }
    
    # User profile and preferences
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive user profile."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Get user stats
        followers_count = await self.get_followers_count(user_id)
        following_count = await self.get_following_count(user_id)
        posts_count = await self.get_posts_count(user_id)
        videos_count = await self.get_videos_count(user_id)
        
        return {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "display_name": user.display_name,
            "bio": user.bio,
            "avatar_url": user.avatar_url,
            "website": user.website,
            "location": user.location,
            "is_verified": user.is_verified,
            "is_private": user.is_private,
            "followers_count": followers_count,
            "following_count": following_count,
            "posts_count": posts_count,
            "videos_count": videos_count,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat()
        }
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Update preferences
        for key, value in preferences.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        await self.db.commit()
        return True
    
    # Helper methods for user stats
    async def get_followers_count(self, user_id: str) -> int:
        """Get user's followers count."""
        result = await self.db.execute(
            select(Follow).where(Follow.following_id == user_id)
        )
        return len(result.scalars().all())
    
    async def get_following_count(self, user_id: str) -> int:
        """Get user's following count."""
        result = await self.db.execute(
            select(Follow).where(Follow.follower_id == user_id)
        )
        return len(result.scalars().all())
    
    async def get_posts_count(self, user_id: str) -> int:
        """Get user's posts count."""
        result = await self.db.execute(
            select(Post).where(Post.author_id == user_id)
        )
        return len(result.scalars().all())
    
    async def get_videos_count(self, user_id: str) -> int:
        """Get user's videos count."""
        result = await self.db.execute(
            select(Video).where(Video.owner_id == user_id)
        )
        return len(result.scalars().all())
