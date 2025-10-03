"""
Authentication schemas for user management.

This module contains Pydantic schemas for user creation, updates, and authentication.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
import re


class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    display_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must contain only letters, numbers, and underscores')
        return v


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    display_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    privacy_level: Optional[str] = Field(None, pattern='^(public|friends|private)$')


class UserResponse(UserBase):
    """Schema for user response."""
    id: str
    is_active: bool
    is_verified: bool
    is_banned: bool
    is_suspended: bool
    followers_count: int
    following_count: int
    posts_count: int
    videos_count: int
    total_views: int
    total_likes: int
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        orm_mode = True


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for access token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(default=1800, description="Token expiration time in seconds")


class TokenData(BaseModel):
    """Schema for token payload data."""
    user_id: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class PasswordReset(BaseModel):
    """Schema for password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class EmailVerification(BaseModel):
    """Schema for email verification."""
    token: str


class TwoFactorSetup(BaseModel):
    """Schema for 2FA setup response."""
    secret: str
    qr_code_url: str


class TwoFactorVerify(BaseModel):
    """Schema for 2FA verification."""
    code: str = Field(..., min_length=6, max_length=6)
