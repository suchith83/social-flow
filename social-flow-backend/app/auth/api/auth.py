"""
Authentication endpoints.

This module contains all authentication-related API endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import AuthenticationError
from app.core.security import create_access_token, create_refresh_token, verify_password
from app.auth.models.user import User
from app.auth.schemas.auth import Token, TokenData, UserCreate, UserResponse
from app.auth.services.enhanced_auth_service import EnhancedAuthService

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    auth_service = EnhancedAuthService(db)
    user = await auth_service.get_user_by_id(token_data.user_id)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Register a new user."""
    auth_service = EnhancedAuthService(db)
    
    # Check if user already exists
    existing_user = await auth_service.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    existing_user = await auth_service.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )
    
    # Create user
    user = await auth_service.create_user(user_data)
    
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Login user and return access token."""
    auth_service = EnhancedAuthService(db)
    
    # Authenticate user
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    # Update last login
    await auth_service.update_last_login(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Refresh access token using refresh token."""
    auth_service = EnhancedAuthService(db)
    
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user = await auth_service.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    # Create new tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get current user information."""
    return current_user


@router.post("/logout")
async def logout(
    logout_all_devices: bool = False,
    refresh_token: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Logout user and revoke tokens."""
    auth_service = EnhancedAuthService(db)
    
    success = await auth_service.logout_user(
        user_id=str(current_user.id),
        access_token=token,
        refresh_token=refresh_token,
        logout_all_devices=logout_all_devices,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout",
        )
    
    return {"message": "Successfully logged out"}


@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Change user password."""
    auth_service = EnhancedAuthService(db)
    
    # Verify old password
    if not verify_password(old_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect old password",
        )
    
    # Update password
    await auth_service.update_password(current_user.id, new_password)
    
    return {"message": "Password updated successfully"}


@router.post("/forgot-password")
async def forgot_password(
    email: str,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Send password reset email."""
    auth_service = EnhancedAuthService(db)
    
    user = await auth_service.get_user_by_email(email)
    if not user:
        # Don't reveal if email exists
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # Generate reset token and send email
    reset_token = create_access_token(
        data={"sub": str(user.id), "type": "password_reset"},
        expires_delta=timedelta(hours=1),
    )
    
    # TODO: Send email with reset link
    # await email_service.send_password_reset_email(user.email, reset_token)
    
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    token: str,
    new_password: str,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Reset password using reset token."""
    auth_service = EnhancedAuthService(db)
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token",
        )
    
    user = await auth_service.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found",
        )
    
    # Update password
    await auth_service.update_password(user.id, new_password)
    
    return {"message": "Password reset successfully"}


@router.post("/verify-email")
async def verify_email(
    verification_token: str,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Verify user email."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.verify_email(verification_token)
        if success:
            return {"message": "Email verified successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid verification token")
    except Exception:
        raise HTTPException(status_code=500, detail="Email verification failed")


@router.post("/2fa/setup")
async def setup_two_factor(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Setup two-factor authentication and return QR code."""
    try:
        auth_service = EnhancedAuthService(db)
        result = await auth_service.setup_2fa(str(current_user.id))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup 2FA: {str(e)}")


@router.post("/2fa/enable")
async def enable_two_factor(
    token: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Enable two-factor authentication with verification token."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.verify_and_enable_2fa(str(current_user.id), token)
        if success:
            return {"message": "2FA enabled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid 2FA token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable 2FA: {str(e)}")


@router.post("/2fa/verify")
async def verify_two_factor(
    token: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Verify two-factor authentication token."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.verify_2fa_token(str(current_user.id), token)
        if success:
            return {"message": "2FA verified successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid 2FA token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"2FA verification failed: {str(e)}")


@router.post("/2fa/disable")
async def disable_two_factor(
    password: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Disable two-factor authentication."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.disable_2fa(str(current_user.id), password)
        if success:
            return {"message": "2FA disabled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to disable 2FA")
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable 2FA: {str(e)}")


@router.post("/oauth/login")
async def oauth_login(
    provider: str,
    provider_user_id: str,
    provider_email: str,
    provider_name: Optional[str] = None,
    provider_avatar: Optional[str] = None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    token_expires_at: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Login or register with OAuth provider."""
    try:
        auth_service = EnhancedAuthService(db)
        result = await auth_service.oauth_login_or_register(
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            provider_name=provider_name,
            provider_avatar=provider_avatar,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=token_expires_at,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth login failed: {str(e)}")


@router.post("/oauth/link")
async def link_oauth_account(
    provider: str,
    provider_user_id: str,
    provider_email: str,
    provider_name: Optional[str] = None,
    provider_avatar: Optional[str] = None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    token_expires_at: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Link OAuth account to existing user."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.link_oauth_account(
            user_id=str(current_user.id),
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            provider_name=provider_name,
            provider_avatar=provider_avatar,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=token_expires_at,
        )
        return {"message": "OAuth account linked successfully" if success else "Failed to link OAuth account"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link OAuth account: {str(e)}")


@router.post("/oauth/unlink")
async def unlink_oauth_account(
    provider: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Unlink OAuth account from user."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.unlink_oauth_account(str(current_user.id), provider)
        return {"message": "OAuth account unlinked successfully" if success else "Failed to unlink OAuth account"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unlink OAuth account: {str(e)}")


@router.get("/profile", response_model=dict)
async def get_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get comprehensive user profile."""
    try:
        auth_service = EnhancedAuthService(db)
        profile = await auth_service.get_user_profile(str(current_user.id))
        if profile:
            return profile
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get user profile")


@router.put("/preferences")
async def update_user_preferences(
    preferences: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update user preferences."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.update_user_preferences(str(current_user.id), preferences)
        if success:
            return {"message": "Preferences updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update preferences")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.post("/sessions/revoke-all")
async def revoke_all_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Revoke all user sessions except current."""
    try:
        auth_service = EnhancedAuthService(db)
        success = await auth_service.revoke_all_user_sessions(str(current_user.id))
        if success:
            return {"message": "All sessions revoked successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to revoke sessions")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to revoke sessions: {str(e)}")


@router.get("/sessions")
async def get_active_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get all active sessions for the current user."""
    try:
        auth_service = EnhancedAuthService(db)
        sessions = await auth_service.get_user_active_sessions(str(current_user.id))
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")
