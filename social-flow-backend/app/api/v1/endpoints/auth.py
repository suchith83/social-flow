"""
Authentication endpoints for user registration, login, and token management.
"""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
    verify_token_type,
    create_email_verification_token,
    create_password_reset_token,
    generate_2fa_secret,
    verify_2fa_token,
    generate_2fa_qr_uri,
)
from app.infrastructure.crud import user as crud_user
from app.models.user import User
from app.schemas.user import (
    UserRegister,
    UserResponse,
    Token,
    TokenRefresh,
    UserLogin,
    TwoFactorSetup,
    TwoFactorVerify,
    TwoFactorDisable,
)
from app.schemas.base import SuccessResponse

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserRegister,
) -> Any:
    """
    Register a new user account.
    
    - **email**: Valid email address (must be unique)
    - **username**: Username (must be unique, 3-50 characters)
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit)
    - **full_name**: User's full name (optional)
    """
    # Check if email already exists
    if await crud_user.is_email_taken(db, email=user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Check if username already exists
    if await crud_user.is_username_taken(db, username=user_in.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )
    
    # Create user (CRUD layer will handle password hashing)
    user = await crud_user.create(db, obj_in=user_in)
    
    # TODO: Send verification email
    # await send_verification_email(user.email, create_email_verification_token(user.id))
    
    return user


@router.post("/login", response_model=Token)
async def login(
    *,
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login.
    
    Get an access token for future requests using username/email and password.
    """
    # Authenticate user (supports email or username)
    user = await crud_user.get_by_email_or_username(db, identifier=form_data.username)
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if 2FA is enabled
    if user.two_factor_enabled and user.two_factor_secret:
        # Return a special token indicating 2FA is required
        temp_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=5),
            additional_claims={"requires_2fa": True},
        )
        return {
            "access_token": temp_token,
            "refresh_token": "",  # Empty string instead of None to satisfy validation
            "token_type": "bearer",
            "requires_2fa": True,
            "expires_in": 300,  # 5 minutes for temp token
        }
    
    # Update last login
    await crud_user.update_last_login(db, user_id=user.id)
    
    # Create tokens
    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes
    }


@router.post("/login/json", response_model=Token)
async def login_json(
    *,
    db: AsyncSession = Depends(get_db),
    credentials: UserLogin,
) -> Any:
    """
    JSON-based login (alternative to OAuth2 form).
    
    - **username_or_email**: User's email or username
    - **password**: User's password
    """
    # Authenticate user
    user = await crud_user.get_by_email_or_username(db, identifier=credentials.username_or_email)
    
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
        )
    
    # Check if 2FA is enabled
    if user.two_factor_enabled and user.two_factor_secret:
        temp_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=5),
            additional_claims={"requires_2fa": True},
        )
        return {
            "access_token": temp_token,
            "refresh_token": None,
            "token_type": "bearer",
            "requires_2fa": True,
        }
    
    # Update last login
    await crud_user.update_last_login(db, user_id=user.id)
    
    # Create tokens
    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    *,
    db: AsyncSession = Depends(get_db),
    token_data: TokenRefresh,
) -> Any:
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    """
    try:
        payload = decode_token(token_data.refresh_token)
        
        # Verify it's a refresh token
        if not verify_token_type(payload, "refresh"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Verify user still exists and is active
    from uuid import UUID
    user = await crud_user.get(db, UUID(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    from app.models.user import UserStatus
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not active",
        )
    
    # Create new tokens
    access_token = create_access_token(subject=user.id)
    new_refresh_token = create_refresh_token(subject=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes
    }


@router.post("/2fa/setup", response_model=TwoFactorSetup)
async def setup_2fa(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Set up two-factor authentication for current user.
    
    Returns QR code URI and secret for authenticator app setup.
    """
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled",
        )
    
    # Generate new secret
    secret = generate_2fa_secret()
    qr_uri = generate_2fa_qr_uri(secret, current_user.email)
    
    # Generate backup codes
    import secrets
    backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
    
    # Store secret temporarily (user must verify before enabling)
    from app.schemas.user import UserUpdate
    await crud_user.update(
        db,
        db_obj=current_user,
        obj_in=UserUpdate(two_factor_secret=secret),
    )
    
    return {
        "secret": secret,
        "qr_code_url": qr_uri,  # Match schema field name
        "backup_codes": backup_codes,
    }


@router.post("/2fa/verify", response_model=SuccessResponse)
async def verify_2fa_setup(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    verification: TwoFactorVerify,
) -> Any:
    """
    Verify 2FA token and enable two-factor authentication.
    
    - **token**: 6-digit code from authenticator app
    """
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled",
        )
    
    if not current_user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA setup not initiated. Call /2fa/setup first",
        )
    
    # Verify token
    if not verify_2fa_token(current_user.two_factor_secret, verification.token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 2FA token",
        )
    
    # Enable 2FA
    from app.schemas.user import UserUpdate
    await crud_user.update(
        db,
        db_obj=current_user,
        obj_in=UserUpdate(two_factor_enabled=True),
    )
    
    return {"success": True, "message": "Two-factor authentication enabled"}


@router.post("/2fa/login", response_model=Token)
async def login_2fa(
    *,
    db: AsyncSession = Depends(get_db),
    token: str = Body(..., embed=True),
    two_fa_token: str = Body(..., embed=True),
) -> Any:
    """
    Complete login with 2FA token.
    
    - **token**: Temporary token from initial login
    - **two_fa_token**: 6-digit code from authenticator app
    """
    try:
        payload = decode_token(token)
        
        # Verify token requires 2FA
        if not payload.get("requires_2fa"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This token does not require 2FA",
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    # Get user
    from uuid import UUID
    user = await crud_user.get(db, UUID(user_id))
    if not user or not user.two_factor_enabled or not user.two_factor_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user or 2FA not enabled",
        )
    
    # Verify 2FA token
    if not verify_2fa_token(user.two_factor_secret, two_fa_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid 2FA token",
        )
    
    # Update last login
    await crud_user.update_last_login(db, user_id=user.id)
    
    # Create full tokens
    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1800,  # 30 minutes
    }


@router.post("/2fa/disable", response_model=SuccessResponse)
async def disable_2fa(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    data: TwoFactorDisable,
) -> Any:
    """
    Disable two-factor authentication.
    
    Requires password verification for security.
    
    - **password**: Current password
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled",
        )
    
    # Verify password
    if not verify_password(data.password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )
    
    # Disable 2FA
    from app.schemas.user import UserUpdate
    await crud_user.update(
        db,
        db_obj=current_user,
        obj_in=UserUpdate(
            two_factor_enabled=False,
            two_factor_secret=None,
        ),
    )
    
    return {"success": True, "message": "Two-factor authentication disabled"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get current user information.
    
    Returns the authenticated user's profile data.
    """
    return current_user
