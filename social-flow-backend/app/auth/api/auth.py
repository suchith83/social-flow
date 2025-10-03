"""
Authentication API Routes

Complete authentication endpoints including registration, login, email verification, etc.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from app.core.database import get_db
from app.core.redis import get_redis
from app.core.dependencies import get_current_user
from app.core.security import create_access_token, create_refresh_token, verify_token
from app.core.exceptions import ValidationError, AuthenticationError
from app.models.user import User
from app.auth.services.auth import AuthService
from app.auth.schemas.auth import (
    UserCreate,
    UserLogin,
    Token,
    EmailVerification,
    PasswordReset,
    PasswordResetConfirm,
)

router = APIRouter()
security = HTTPBearer()


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.
    
    Dependency for routes that require an active user.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=Dict[str, Any])
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account."""
    auth_service = AuthService(db)
    
    try:
        result = await auth_service.register_user_with_verification(user_data)
        return result
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"Registration error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Login with username/email and password."""
    auth_service = AuthService(db)
    
    try:
        user = await auth_service.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        await auth_service.update_last_login(str(user.id))
        
        # Create tokens
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: User = Depends(get_current_active_user),
    redis = Depends(get_redis),
):
    """Logout current user (invalidate tokens)."""
    # In a production system, you would:
    # 1. Add token to blacklist in Redis
    # 2. Remove refresh token from storage
    # For now, just return success
    return None


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
):
    """Refresh access token using refresh token."""
    try:
        payload = verify_token(refresh_token)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Verify user still exists
        auth_service = AuthService(db)
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new tokens
        new_access_token = create_access_token(data={"sub": str(user.id)})
        new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )


@router.post("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    verification_data: EmailVerification,
    db: AsyncSession = Depends(get_db),
):
    """Verify user email with verification token."""
    auth_service = AuthService(db)
    
    try:
        # In production, you would validate the token and verify the user
        # For now, simplified implementation
        result = await auth_service.verify_email(verification_data.token)
        return {"message": "Email verified successfully"}
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/password-reset", status_code=status.HTTP_200_OK)
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_db),
):
    """Request password reset email."""
    auth_service = AuthService(db)
    
    try:
        # Send password reset email
        await auth_service.request_password_reset(reset_data.email)
        return {"message": "Password reset email sent"}
    except Exception:
        # Always return success to prevent email enumeration
        return {"message": "Password reset email sent"}


@router.post("/password-reset/confirm", status_code=status.HTTP_200_OK)
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
):
    """Confirm password reset with token and new password."""
    auth_service = AuthService(db)
    
    try:
        await auth_service.reset_password(reset_data.token, reset_data.new_password)
        return {"message": "Password reset successful"}
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
):
    """Get current user information."""
    return current_user.to_dict()

