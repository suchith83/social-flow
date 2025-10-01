"""
Enhanced authentication schemas.

Pydantic schemas for enhanced auth features including 2FA, OAuth, and RBAC.
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# ============================================================================
# TOKEN SCHEMAS
# ============================================================================

class TokenPair(BaseModel):
    """Token pair response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request schema."""
    refresh_token: str = Field(..., min_length=1)
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class TokenRevoke(BaseModel):
    """Token revocation request schema."""
    access_token: str = Field(..., min_length=1)
    refresh_token: Optional[str] = None
    logout_all_devices: bool = False


# ============================================================================
# TWO-FACTOR AUTHENTICATION SCHEMAS
# ============================================================================

class TwoFactorSetup(BaseModel):
    """2FA setup response schema."""
    secret: str
    qr_code: str
    backup_codes: List[str] = []
    message: str


class TwoFactorVerify(BaseModel):
    """2FA verification request schema."""
    token: str = Field(..., min_length=6, max_length=6)


class TwoFactorEnable(BaseModel):
    """2FA enable request schema."""
    token: str = Field(..., min_length=6, max_length=6)


class TwoFactorDisable(BaseModel):
    """2FA disable request schema."""
    password: str = Field(..., min_length=1)


class TwoFactorLogin(BaseModel):
    """2FA login request schema."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    totp_token: Optional[str] = Field(None, min_length=6, max_length=6)


# ============================================================================
# OAUTH SOCIAL LOGIN SCHEMAS
# ============================================================================

class OAuthProvider(BaseModel):
    """OAuth provider enum."""
    provider: str = Field(..., description="OAuth provider (google, facebook, apple)")


class OAuthLogin(BaseModel):
    """OAuth login request schema."""
    provider: str = Field(..., description="OAuth provider name")
    code: Optional[str] = Field(None, description="Authorization code from OAuth provider")
    access_token: Optional[str] = Field(None, description="Access token from OAuth provider")
    id_token: Optional[str] = Field(None, description="ID token from OAuth provider")


class OAuthCallback(BaseModel):
    """OAuth callback data schema."""
    provider: str
    provider_user_id: str
    provider_email: str
    provider_name: Optional[str] = None
    provider_avatar: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None


class OAuthAccountResponse(BaseModel):
    """OAuth account response schema."""
    id: str
    provider: str
    provider_email: Optional[str]
    provider_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class OAuthLink(BaseModel):
    """OAuth account link request schema."""
    provider: str
    code: Optional[str] = None
    access_token: Optional[str] = None


class OAuthUnlink(BaseModel):
    """OAuth account unlink request schema."""
    provider: str


# ============================================================================
# ROLE-BASED ACCESS CONTROL SCHEMAS
# ============================================================================

class PermissionBase(BaseModel):
    """Base permission schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    resource: str = Field(..., min_length=1, max_length=50)
    action: str = Field(..., min_length=1, max_length=50)


class PermissionCreate(PermissionBase):
    """Permission creation schema."""
    pass


class PermissionResponse(PermissionBase):
    """Permission response schema."""
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RoleBase(BaseModel):
    """Base role schema."""
    name: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    priority: str = Field(default="0")


class RoleCreate(RoleBase):
    """Role creation schema."""
    permission_ids: List[str] = []


class RoleUpdate(BaseModel):
    """Role update schema."""
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    priority: Optional[str] = None
    permission_ids: Optional[List[str]] = None


class RoleResponse(RoleBase):
    """Role response schema."""
    id: str
    is_active: bool
    is_system: bool
    created_at: datetime
    updated_at: datetime
    permissions: List[PermissionResponse] = []
    
    class Config:
        from_attributes = True


class RoleAssignment(BaseModel):
    """Role assignment request schema."""
    user_id: str = Field(..., description="User ID to assign role to")
    role_name: str = Field(..., description="Role name to assign")


class RoleRemoval(BaseModel):
    """Role removal request schema."""
    user_id: str = Field(..., description="User ID to remove role from")
    role_name: str = Field(..., description="Role name to remove")


class PermissionCheck(BaseModel):
    """Permission check request schema."""
    user_id: str = Field(..., description="User ID to check")
    permission_name: str = Field(..., description="Permission name to check")


class UserRolesResponse(BaseModel):
    """User roles response schema."""
    user_id: str
    roles: List[RoleResponse]
    permissions: List[str]


# ============================================================================
# ENHANCED LOGIN/REGISTER SCHEMAS
# ============================================================================

class EnhancedLoginRequest(BaseModel):
    """Enhanced login request with 2FA support."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    totp_token: Optional[str] = Field(None, min_length=6, max_length=6)
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class EnhancedLoginResponse(BaseModel):
    """Enhanced login response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    requires_2fa: bool = False
    user: dict


class EnhancedRegisterRequest(BaseModel):
    """Enhanced registration request."""
    username: str = Field(..., min_length=3, max_length=20)
    email: str = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)


class EnhancedRegisterResponse(BaseModel):
    """Enhanced registration response."""
    user_id: str
    username: str
    email: str
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    message: str = "Registration successful"


# ============================================================================
# SESSION MANAGEMENT SCHEMAS
# ============================================================================

class ActiveSession(BaseModel):
    """Active session information."""
    session_id: str
    device_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    expires_at: datetime
    last_used_at: Optional[datetime]
    is_current: bool


class SessionList(BaseModel):
    """List of active sessions."""
    sessions: List[ActiveSession]


class SessionRevoke(BaseModel):
    """Session revocation request."""
    session_id: str


class SessionRevokeAll(BaseModel):
    """Revoke all sessions except current."""
    keep_current: bool = True
