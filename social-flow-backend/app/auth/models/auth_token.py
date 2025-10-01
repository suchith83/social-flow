"""
Authentication token models.

This module defines models for refresh tokens, token blacklist, and OAuth tokens.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class EmailVerificationToken(Base):
    """Email verification token model."""
    
    __tablename__ = "email_verification_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), nullable=False)
    
    # Token status
    is_used = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", backref="verification_tokens")
    
    def __repr__(self) -> str:
        return f"<EmailVerificationToken(id={self.id}, user_id={self.user_id}, is_used={self.is_used})>"
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid."""
        if self.is_used:
            return False
        if self.expires_at < datetime.utcnow():
            return False
        return True


class PasswordResetToken(Base):
    """Password reset token model."""
    
    __tablename__ = "password_reset_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), nullable=False)
    
    # Token status
    is_used = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", backref="password_reset_tokens")
    
    def __repr__(self) -> str:
        return f"<PasswordResetToken(id={self.id}, user_id={self.user_id}, is_used={self.is_used})>"
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid."""
        if self.is_used:
            return False
        if self.expires_at < datetime.utcnow():
            return False
        return True


class RefreshToken(Base):
    """Refresh token model for JWT token rotation."""
    
    __tablename__ = "refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(500), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Token metadata
    token_family = Column(String(255), nullable=False, index=True)  # For token rotation tracking
    device_id = Column(String(255), nullable=True)  # Device fingerprint
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Token status
    is_revoked = Column(Boolean, default=False, nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", backref="refresh_tokens")
    
    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid."""
        if self.is_revoked or self.is_used:
            return False
        if self.expires_at < datetime.utcnow():
            return False
        return True


class TokenBlacklist(Base):
    """Token blacklist for revoked access tokens."""
    
    __tablename__ = "token_blacklist"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID
    token = Column(Text, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Revocation details
    reason = Column(String(100), nullable=True)  # logout, password_change, security_breach, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", backref="blacklisted_tokens")
    
    def __repr__(self) -> str:
        return f"<TokenBlacklist(jti={self.jti}, user_id={self.user_id}, reason={self.reason})>"


class OAuthAccount(Base):
    """OAuth account for social login."""
    
    __tablename__ = "oauth_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # OAuth provider details
    provider = Column(String(50), nullable=False, index=True)  # google, facebook, apple, twitter
    provider_user_id = Column(String(255), nullable=False, index=True)
    provider_email = Column(String(255), nullable=True)
    provider_name = Column(String(255), nullable=True)
    provider_avatar = Column(String(500), nullable=True)
    
    # OAuth tokens
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", backref="oauth_accounts")
    
    def __repr__(self) -> str:
        return f"<OAuthAccount(id={self.id}, provider={self.provider}, user_id={self.user_id})>"


class TwoFactorAuth(Base):
    """Two-factor authentication model."""
    
    __tablename__ = "two_factor_auth"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    
    # TOTP details
    secret = Column(String(255), nullable=False)  # Encrypted TOTP secret
    is_enabled = Column(Boolean, default=False, nullable=False)
    
    # Backup codes (encrypted, comma-separated)
    backup_codes = Column(Text, nullable=True)
    backup_codes_used = Column(Integer, default=0, nullable=False)
    
    # Verification status
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    enabled_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", backref="two_factor_auth", uselist=False)
    
    def __repr__(self) -> str:
        return f"<TwoFactorAuth(id={self.id}, user_id={self.user_id}, is_enabled={self.is_enabled})>"
