"""
User Pydantic schemas for API request/response validation.

This module provides schemas for user-related operations including:
- User registration and authentication
- Profile management
- OAuth integration
- 2FA setup
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import ConfigDict, EmailStr, Field, field_validator

from app.schemas.base import BaseDBSchema, BaseSchema


# Enums (matching models)
class UserRole(str):
    """User role enumeration."""
    USER = "user"
    CREATOR = "creator"
    MODERATOR = "moderator"
    ADMIN = "admin"


class UserStatus(str):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"


class OAuthProvider(str):
    """OAuth provider enumeration."""
    GOOGLE = "google"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    APPLE = "apple"
    GITHUB = "github"


# Base User schemas
class UserBase(BaseSchema):
    """Base user schema with common fields."""
    
    username: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$', 
                         description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    cover_image_url: Optional[str] = Field(None, description="Cover image URL")
    website_url: Optional[str] = Field(None, description="Personal website URL")
    location: Optional[str] = Field(None, max_length=100, description="User location")


# Create schemas
class UserCreate(UserBase):
    """Schema for user registration."""
    
    password: str = Field(..., min_length=8, max_length=100, description="User password (min 8 characters)")
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserRegister(BaseSchema):
    """Simplified registration schema."""
    
    username: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    display_name: Optional[str] = Field(None, max_length=255)
    full_name: Optional[str] = Field(None, max_length=255)  # Deprecated, use display_name
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


# Update schemas
class UserUpdate(BaseSchema):
    """Schema for updating user profile."""
    
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    website_url: Optional[str] = None
    location: Optional[str] = Field(None, max_length=100)
    is_verified: Optional[bool] = None
    is_private: Optional[bool] = None


class UserUpdatePassword(BaseSchema):
    """Schema for password change."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


# Response schemas
class UserResponse(BaseDBSchema):
    """User response schema (public data)."""
    
    username: str
    email: EmailStr
    display_name: Optional[str] = None
    full_name: Optional[str] = None  # Deprecated, use display_name
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    website_url: Optional[str] = None
    location: Optional[str] = None
    role: str
    status: str
    is_verified: bool
    is_private: bool
    
    # Social stats
    followers_count: int = 0
    following_count: int = 0
    videos_count: int = 0
    posts_count: int = 0
    
    # Timestamps
    last_login_at: Optional[datetime] = None
    email_verified_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserDetailResponse(UserResponse):
    """Detailed user response (for authenticated user viewing their own profile)."""
    
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    
    # Account status
    is_email_verified: bool = False
    is_phone_verified: bool = False
    is_2fa_enabled: bool = False
    
    # Stripe integration
    stripe_customer_id: Optional[str] = None
    stripe_connect_account_id: Optional[str] = None
    
    # Preferences
    notification_preferences: dict = {}


class UserPublicResponse(BaseDBSchema):
    """Public user profile (minimal data)."""
    
    username: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    website_url: Optional[str] = None
    location: Optional[str] = None
    is_verified: bool
    
    # Social stats
    followers_count: int = 0
    following_count: int = 0
    videos_count: int = 0
    posts_count: int = 0


# Authentication schemas
class UserLogin(BaseSchema):
    """Login credentials."""
    
    username_or_email: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class Token(BaseSchema):
    """JWT token response."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class TokenRefresh(BaseSchema):
    """Token refresh request."""
    
    refresh_token: str = Field(..., description="Refresh token")


# OAuth schemas
class OAuthLogin(BaseSchema):
    """OAuth login request."""
    
    provider: str = Field(..., description="OAuth provider (google, facebook, etc.)")
    access_token: str = Field(..., description="OAuth access token from provider")


class OAuthConnect(BaseSchema):
    """Connect OAuth account to existing user."""
    
    provider: str = Field(..., description="OAuth provider")
    provider_user_id: str = Field(..., description="User ID from OAuth provider")
    access_token: Optional[str] = Field(None, description="OAuth access token")


# 2FA schemas
class TwoFactorSetup(BaseSchema):
    """2FA setup response."""
    
    secret: str = Field(..., description="TOTP secret key")
    qr_code_url: str = Field(..., description="QR code URL for authenticator app")
    backup_codes: list[str] = Field(..., description="Backup codes for recovery")


class TwoFactorVerify(BaseSchema):
    """2FA verification request."""
    
    code: str = Field(..., min_length=6, max_length=6, pattern=r'^\d{6}$', description="6-digit TOTP code")


class TwoFactorDisable(BaseSchema):
    """Disable 2FA request."""
    
    password: str = Field(..., description="User password for verification")
    code: Optional[str] = Field(None, description="Current 2FA code (if available)")


# Admin schemas
class UserAdminUpdate(BaseSchema):
    """Admin-only user update schema."""
    
    role: Optional[str] = None
    status: Optional[str] = None
    is_verified: Optional[bool] = None
    suspension_reason: Optional[str] = None
    suspension_expires_at: Optional[datetime] = None


# List and filter schemas
class UserListFilters(BaseSchema):
    """Filters for user listing."""
    
    role: Optional[str] = None
    status: Optional[str] = None
    is_verified: Optional[bool] = None
    search: Optional[str] = Field(None, description="Search by username, email, or full name")
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
