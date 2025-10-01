"""
Enhanced authentication service with JWT rotation, 2FA, OAuth, and RBAC.

This module provides comprehensive authentication features including:
- JWT access & refresh token management with rotation
- Token revocation and blacklisting
- Two-factor authentication (TOTP)
- OAuth social login (Google, Facebook, Apple)
- Role-based access control (RBAC)
"""

import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

import pyotp
import qrcode
from io import BytesIO
import base64

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_password_hash,
    verify_password,
)
from app.core.exceptions import AuthenticationError, ValidationError
from app.auth.models.user import User
from app.auth.models.auth_token import RefreshToken, TokenBlacklist, OAuthAccount, TwoFactorAuth
from app.auth.models.rbac import Role


class EnhancedAuthService:
    """Enhanced authentication service with advanced security features."""
    
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
    
    # ============================================================================
    # JWT TOKEN MANAGEMENT WITH ROTATION
    # ============================================================================
    
    async def create_token_pair(
        self,
        user: User,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create access and refresh token pair with rotation support."""
        # Generate unique JWT ID for access token
        jti = str(uuid.uuid4())
        
        # Create access token
        access_token_data = {
            "sub": str(user.id),
            "username": user.username,
            "jti": jti,
            "type": "access",
        }
        access_token = create_access_token(access_token_data)
        
        # Create refresh token with token family for rotation tracking
        token_family = str(uuid.uuid4())
        refresh_token_data = {
            "sub": str(user.id),
            "token_family": token_family,
            "type": "refresh",
        }
        refresh_token_str = create_refresh_token(refresh_token_data)
        
        # Store refresh token in database
        refresh_token = RefreshToken(
            token=refresh_token_str,
            user_id=user.id,
            token_family=token_family,
            device_id=device_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        
        self.db.add(refresh_token)
        await self.db.commit()
        
        # Cache access token JTI in Redis for quick revocation checks
        await self.redis.setex(
            f"access_token:{jti}",
            settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            json.dumps({
                "user_id": str(user.id),
                "username": user.username,
                "created_at": datetime.utcnow().isoformat(),
            })
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token_str,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }
    
    async def rotate_refresh_token(
        self,
        refresh_token: str,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rotate refresh token - creates new token pair and revokes old refresh token."""
        # Verify refresh token
        payload = verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token")
        
        # Get refresh token from database
        result = await self.db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.token == refresh_token,
                    RefreshToken.is_revoked == False,
                    RefreshToken.is_used == False,
                )
            )
        )
        db_refresh_token = result.scalar_one_or_none()
        
        if not db_refresh_token:
            # Token reuse detected - revoke entire token family
            token_family = payload.get("token_family")
            if token_family:
                await self.revoke_token_family(token_family, reason="token_reuse_detected")
            raise AuthenticationError("Refresh token reuse detected. All tokens revoked for security.")
        
        if not db_refresh_token.is_valid:
            raise AuthenticationError("Refresh token is invalid or expired")
        
        # Get user
        user = await self.db.get(User, db_refresh_token.user_id)
        if not user or not user.is_authenticated:
            raise AuthenticationError("User is not authenticated")
        
        # Mark old refresh token as used
        db_refresh_token.is_used = True
        db_refresh_token.used_at = datetime.utcnow()
        await self.db.commit()
        
        # Create new token pair
        new_tokens = await self.create_token_pair(
            user=user,
            device_id=device_id or db_refresh_token.device_id,
            ip_address=ip_address or db_refresh_token.ip_address,
            user_agent=user_agent or db_refresh_token.user_agent,
        )
        
        return new_tokens
    
    async def revoke_token_family(self, token_family: str, reason: str = "security"):
        """Revoke all refresh tokens in a token family (for security breach detection)."""
        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_family == token_family)
        )
        tokens = result.scalars().all()
        
        for token in tokens:
            token.is_revoked = True
            token.revoked_at = datetime.utcnow()
        
        await self.db.commit()
        
        # Also add to blacklist if needed
        for token in tokens:
            await self.blacklist_token(
                token=token.token,
                user_id=str(token.user_id),
                reason=reason,
            )
    
    async def is_token_blacklisted(self, jti: str) -> bool:
        """Check if access token is blacklisted."""
        # Check Redis cache first (fast)
        blacklisted = await self.redis.get(f"blacklist:{jti}")
        if blacklisted:
            return True
        
        # Check database
        result = await self.db.execute(
            select(TokenBlacklist).where(TokenBlacklist.jti == jti)
        )
        return result.scalar_one_or_none() is not None
    
    async def blacklist_token(
        self,
        token: str,
        user_id: str,
        reason: str = "logout",
    ) -> bool:
        """Add access token to blacklist."""
        payload = verify_token(token)
        if not payload:
            return False
        
        jti = payload.get("jti")
        if not jti:
            return False
        
        # Calculate expiration
        exp_timestamp = payload.get("exp")
        expires_at = datetime.fromtimestamp(exp_timestamp)
        
        # Add to database
        blacklist_entry = TokenBlacklist(
            jti=jti,
            token=token,
            user_id=uuid.UUID(user_id),
            reason=reason,
            expires_at=expires_at,
        )
        
        self.db.add(blacklist_entry)
        await self.db.commit()
        
        # Cache in Redis for quick lookups
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        if ttl > 0:
            await self.redis.setex(f"blacklist:{jti}", ttl, "1")
        
        return True
    
    async def logout_user(
        self,
        user_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        logout_all_devices: bool = False,
    ) -> bool:
        """Logout user by revoking tokens."""
        # Blacklist current access token
        await self.blacklist_token(access_token, user_id, reason="logout")
        
        if logout_all_devices:
            # Revoke all user's refresh tokens
            result = await self.db.execute(
                select(RefreshToken).where(
                    and_(
                        RefreshToken.user_id == uuid.UUID(user_id),
                        RefreshToken.is_revoked == False,
                    )
                )
            )
            tokens = result.scalars().all()
            
            for token in tokens:
                token.is_revoked = True
                token.revoked_at = datetime.utcnow()
            
            await self.db.commit()
        elif refresh_token:
            # Revoke only current refresh token
            result = await self.db.execute(
                select(RefreshToken).where(RefreshToken.token == refresh_token)
            )
            db_token = result.scalar_one_or_none()
            
            if db_token:
                db_token.is_revoked = True
                db_token.revoked_at = datetime.utcnow()
                await self.db.commit()
        
        return True
    
    # ============================================================================
    # TWO-FACTOR AUTHENTICATION (TOTP)
    # ============================================================================
    
    async def setup_2fa(self, user_id: str) -> Dict[str, Any]:
        """Initialize 2FA setup for user."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user:
            raise ValidationError("User not found")
        
        # Check if 2FA already exists
        result = await self.db.execute(
            select(TwoFactorAuth).where(TwoFactorAuth.user_id == user.id)
        )
        existing_2fa = result.scalar_one_or_none()
        
        if existing_2fa and existing_2fa.is_enabled:
            raise ValidationError("2FA is already enabled for this user")
        
        # Generate TOTP secret
        secret = pyotp.random_base32()
        
        # Create or update 2FA record
        if existing_2fa:
            existing_2fa.secret = secret
            existing_2fa.is_enabled = False
            existing_2fa.is_verified = False
            existing_2fa.updated_at = datetime.utcnow()
            two_fa = existing_2fa
        else:
            # Generate backup codes
            backup_codes = [secrets.token_hex(4) for _ in range(10)]
            backup_codes_str = ",".join([get_password_hash(code) for code in backup_codes])
            
            two_fa = TwoFactorAuth(
                user_id=user.id,
                secret=secret,
                backup_codes=backup_codes_str,
                is_enabled=False,
                is_verified=False,
            )
            self.db.add(two_fa)
        
        await self.db.commit()
        
        # Generate QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user.email,
            issuer_name="SocialFlow"
        )
        
        # Create QR code image
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "secret": secret,
            "qr_code": f"data:image/png;base64,{qr_code_base64}",
            "backup_codes": backup_codes if not existing_2fa else [],
            "message": "Scan the QR code with your authenticator app and verify with a code to enable 2FA"
        }
    
    async def verify_and_enable_2fa(self, user_id: str, token: str) -> bool:
        """Verify TOTP token and enable 2FA."""
        result = await self.db.execute(
            select(TwoFactorAuth).where(TwoFactorAuth.user_id == uuid.UUID(user_id))
        )
        two_fa = result.scalar_one_or_none()
        
        if not two_fa:
            raise ValidationError("2FA not set up for this user")
        
        # Verify token
        totp = pyotp.TOTP(two_fa.secret)
        if not totp.verify(token, valid_window=1):
            raise AuthenticationError("Invalid 2FA token")
        
        # Enable 2FA
        two_fa.is_enabled = True
        two_fa.is_verified = True
        two_fa.enabled_at = datetime.utcnow()
        two_fa.updated_at = datetime.utcnow()
        
        await self.db.commit()
        
        return True
    
    async def verify_2fa_token(self, user_id: str, token: str) -> bool:
        """Verify 2FA TOTP token during login."""
        result = await self.db.execute(
            select(TwoFactorAuth).where(
                and_(
                    TwoFactorAuth.user_id == uuid.UUID(user_id),
                    TwoFactorAuth.is_enabled == True,
                )
            )
        )
        two_fa = result.scalar_one_or_none()
        
        if not two_fa:
            return True  # 2FA not enabled
        
        # Verify TOTP token
        totp = pyotp.TOTP(two_fa.secret)
        if totp.verify(token, valid_window=1):
            two_fa.last_used_at = datetime.utcnow()
            await self.db.commit()
            return True
        
        # Try backup codes
        if two_fa.backup_codes:
            backup_codes = two_fa.backup_codes.split(",")
            for hashed_code in backup_codes:
                if verify_password(token, hashed_code):
                    # Remove used backup code
                    backup_codes.remove(hashed_code)
                    two_fa.backup_codes = ",".join(backup_codes)
                    two_fa.backup_codes_used += 1
                    two_fa.last_used_at = datetime.utcnow()
                    await self.db.commit()
                    return True
        
        return False
    
    async def disable_2fa(self, user_id: str, password: str) -> bool:
        """Disable 2FA after password verification."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user or not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid password")
        
        result = await self.db.execute(
            select(TwoFactorAuth).where(TwoFactorAuth.user_id == user.id)
        )
        two_fa = result.scalar_one_or_none()
        
        if two_fa:
            two_fa.is_enabled = False
            two_fa.updated_at = datetime.utcnow()
            await self.db.commit()
        
        return True
    
    # ============================================================================
    # OAUTH SOCIAL LOGIN
    # ============================================================================
    
    async def link_oauth_account(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: Optional[str] = None,
        provider_name: Optional[str] = None,
        provider_avatar: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_expires_at: Optional[datetime] = None,
    ) -> OAuthAccount:
        """Link OAuth account to user."""
        # Check if OAuth account already exists
        result = await self.db.execute(
            select(OAuthAccount).where(
                and_(
                    OAuthAccount.provider == provider,
                    OAuthAccount.provider_user_id == provider_user_id,
                )
            )
        )
        existing_oauth = result.scalar_one_or_none()
        
        if existing_oauth:
            if str(existing_oauth.user_id) != user_id:
                raise ValidationError(f"This {provider} account is already linked to another user")
            
            # Update existing OAuth account
            existing_oauth.provider_email = provider_email
            existing_oauth.provider_name = provider_name
            existing_oauth.provider_avatar = provider_avatar
            existing_oauth.access_token = access_token
            existing_oauth.refresh_token = refresh_token
            existing_oauth.token_expires_at = token_expires_at
            existing_oauth.updated_at = datetime.utcnow()
            existing_oauth.last_used_at = datetime.utcnow()
            await self.db.commit()
            
            return existing_oauth
        
        # Create new OAuth account
        oauth_account = OAuthAccount(
            user_id=uuid.UUID(user_id),
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            provider_name=provider_name,
            provider_avatar=provider_avatar,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=token_expires_at,
            last_used_at=datetime.utcnow(),
        )
        
        self.db.add(oauth_account)
        await self.db.commit()
        await self.db.refresh(oauth_account)
        
        return oauth_account
    
    async def oauth_login_or_register(
        self,
        provider: str,
        provider_user_id: str,
        provider_email: str,
        provider_name: Optional[str] = None,
        provider_avatar: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Login or register user with OAuth."""
        # Check if OAuth account exists
        result = await self.db.execute(
            select(OAuthAccount).where(
                and_(
                    OAuthAccount.provider == provider,
                    OAuthAccount.provider_user_id == provider_user_id,
                )
            )
        )
        oauth_account = result.scalar_one_or_none()
        
        if oauth_account:
            # Existing user - login
            user = await self.db.get(User, oauth_account.user_id)
            if not user or not user.is_authenticated:
                raise AuthenticationError("User account is not active")
            
            # Update OAuth account
            oauth_account.last_used_at = datetime.utcnow()
            oauth_account.access_token = access_token
            oauth_account.refresh_token = refresh_token
            oauth_account.token_expires_at = token_expires_at
            await self.db.commit()
            
        else:
            # Check if user exists with this email
            result = await self.db.execute(
                select(User).where(User.email == provider_email)
            )
            user = result.scalar_one_or_none()
            
            if user:
                # Link OAuth account to existing user
                await self.link_oauth_account(
                    user_id=str(user.id),
                    provider=provider,
                    provider_user_id=provider_user_id,
                    provider_email=provider_email,
                    provider_name=provider_name,
                    provider_avatar=provider_avatar,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_expires_at=token_expires_at,
                )
            else:
                # Create new user
                username = self._generate_username_from_email(provider_email)
                user = User(
                    username=username,
                    email=provider_email,
                    password_hash=get_password_hash(secrets.token_urlsafe(32)),  # Random password
                    display_name=provider_name or username,
                    avatar_url=provider_avatar,
                    is_verified=True,  # OAuth users are auto-verified
                )
                
                self.db.add(user)
                await self.db.commit()
                await self.db.refresh(user)
                
                # Link OAuth account
                await self.link_oauth_account(
                    user_id=str(user.id),
                    provider=provider,
                    provider_user_id=provider_user_id,
                    provider_email=provider_email,
                    provider_name=provider_name,
                    provider_avatar=provider_avatar,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_expires_at=token_expires_at,
                )
                
                # Assign default role
                await self.assign_default_role(user)
        
        # Update last login
        user.last_login_at = datetime.utcnow()
        await self.db.commit()
        
        # Create token pair
        tokens = await self.create_token_pair(user)
        
        return {
            **tokens,
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "display_name": user.display_name,
                "avatar_url": user.avatar_url,
                "is_verified": user.is_verified,
            }
        }
    
    def _generate_username_from_email(self, email: str) -> str:
        """Generate unique username from email."""
        base_username = email.split("@")[0].lower()
        base_username = "".join(c if c.isalnum() or c == "_" else "_" for c in base_username)
        
        # Add random suffix to ensure uniqueness
        username = f"{base_username}_{secrets.token_hex(4)}"
        return username[:50]  # Limit to max username length
    
    async def unlink_oauth_account(self, user_id: str, provider: str) -> bool:
        """Unlink OAuth account from user."""
        result = await self.db.execute(
            select(OAuthAccount).where(
                and_(
                    OAuthAccount.user_id == uuid.UUID(user_id),
                    OAuthAccount.provider == provider,
                )
            )
        )
        oauth_account = result.scalar_one_or_none()
        
        if oauth_account:
            await self.db.delete(oauth_account)
            await self.db.commit()
            return True
        
        return False
    
    # ============================================================================
    # ROLE-BASED ACCESS CONTROL (RBAC)
    # ============================================================================
    
    async def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to user."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user:
            raise ValidationError("User not found")
        
        result = await self.db.execute(
            select(Role).where(and_(Role.name == role_name, Role.is_active == True))
        )
        role = result.scalar_one_or_none()
        
        if not role:
            raise ValidationError(f"Role '{role_name}' not found")
        
        # Check if user already has this role
        if not user.has_role(role_name):
            user.roles.append(role)
            await self.db.commit()
        
        return True
    
    async def remove_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Remove role from user."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user:
            raise ValidationError("User not found")
        
        result = await self.db.execute(
            select(Role).where(Role.name == role_name)
        )
        role = result.scalar_one_or_none()
        
        if role and role in user.roles:
            user.roles.remove(role)
            await self.db.commit()
            return True
        
        return False
    
    async def assign_default_role(self, user: User) -> bool:
        """Assign default 'viewer' role to new user."""
        result = await self.db.execute(
            select(Role).where(and_(Role.name == "viewer", Role.is_active == True))
        )
        viewer_role = result.scalar_one_or_none()
        
        if viewer_role and viewer_role not in user.roles:
            user.roles.append(viewer_role)
            await self.db.commit()
            return True
        
        return False
    
    async def check_permission(self, user_id: str, permission_name: str) -> bool:
        """Check if user has specific permission."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user:
            return False
        
        return user.has_permission(permission_name)
    
    async def get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all roles for user."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user:
            return []
        
        return [
            {
                "id": str(role.id),
                "name": role.name,
                "display_name": role.display_name,
                "description": role.description,
            }
            for role in user.roles if role.is_active
        ]
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user."""
        user = await self.db.get(User, uuid.UUID(user_id))
        if not user:
            return []
        
        return user.get_permissions()
    
    async def revoke_all_user_sessions(self, user_id: str) -> bool:
        """Revoke all refresh tokens for user except current session."""
        try:
            # Get all active refresh tokens for user
            result = await self.db.execute(
                select(RefreshToken).where(
                    and_(
                        RefreshToken.user_id == uuid.UUID(user_id),
                        not RefreshToken.is_revoked,
                    )
                )
            )
            tokens = result.scalars().all()
            
            # Revoke all tokens
            for token in tokens:
                token.is_revoked = True
                token.revoked_at = datetime.utcnow()
            
            await self.db.commit()
            return True
        except Exception:
            return False
    
    async def get_user_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for user."""
        try:
            result = await self.db.execute(
                select(RefreshToken).where(
                    and_(
                        RefreshToken.user_id == uuid.UUID(user_id),
                        not RefreshToken.is_revoked,
                    )
                )
            )
            tokens = result.scalars().all()
            
            sessions = []
            for token in tokens:
                sessions.append({
                    "id": str(token.id),
                    "device_id": token.device_id,
                    "ip_address": token.ip_address,
                    "user_agent": token.user_agent,
                    "created_at": token.created_at.isoformat() if token.created_at else None,
                    "expires_at": token.expires_at.isoformat() if token.expires_at else None,
                })
            
            return sessions
        except Exception:
            return []
