"""
Unit tests for Auth Domain Value Objects

Tests validation logic and business rules for Email, Username, Password, etc.
"""

import pytest
from datetime import datetime, timedelta

from app.auth.domain.value_objects import (
    Email,
    Username,
    Password,
    AccountStatus,
    PrivacyLevel,
    SuspensionDetails,
    BanDetails,
)


class TestEmail:
    """Tests for Email value object."""
    
    def test_valid_email(self):
        """Test creating a valid email."""
        email = Email("user@example.com")
        assert email.value == "user@example.com"
        assert email.domain == "example.com"
        assert email.local_part == "user"
    
    def test_email_normalization(self):
        """Test email is normalized to lowercase."""
        email = Email("User@Example.COM")
        assert email.value == "user@example.com"
    
    def test_invalid_email_format(self):
        """Test invalid email formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid email format"):
            Email("invalid-email")
        
        with pytest.raises(ValueError, match="Invalid email format"):
            Email("@example.com")
        
        with pytest.raises(ValueError, match="Invalid email format"):
            Email("user@")
    
    def test_empty_email(self):
        """Test empty email raises ValueError."""
        with pytest.raises(ValueError, match="Email cannot be empty"):
            Email("")
    
    def test_email_too_long(self):
        """Test email exceeding 255 characters raises ValueError."""
        long_email = "a" * 250 + "@example.com"
        with pytest.raises(ValueError, match="cannot exceed 255 characters"):
            Email(long_email)
    
    def test_email_immutable(self):
        """Test email is immutable."""
        email = Email("user@example.com")
        with pytest.raises(AttributeError):
            email.value = "different@example.com"  # type: ignore


class TestUsername:
    """Tests for Username value object."""
    
    def test_valid_username(self):
        """Test creating a valid username."""
        username = Username("john_doe")
        assert username.value == "john_doe"
    
    def test_username_minimum_length(self):
        """Test username must be at least 3 characters."""
        with pytest.raises(ValueError, match="must be at least 3 characters"):
            Username("ab")
    
    def test_username_maximum_length(self):
        """Test username cannot exceed 50 characters."""
        long_username = "a" * 51
        with pytest.raises(ValueError, match="cannot exceed 50 characters"):
            Username(long_username)
    
    def test_username_must_start_with_letter(self):
        """Test username must start with a letter."""
        with pytest.raises(ValueError, match="Invalid username format"):
            Username("123invalid")
        
        with pytest.raises(ValueError, match="Invalid username format"):
            Username("_invalid")
    
    def test_username_valid_characters(self):
        """Test username can contain letters, numbers, and underscores."""
        # Valid usernames
        assert Username("john").value == "john"
        assert Username("john123").value == "john123"
        assert Username("john_doe").value == "john_doe"
        assert Username("JohnDoe").value == "JohnDoe"
    
    def test_username_invalid_characters(self):
        """Test username with invalid characters raises ValueError."""
        with pytest.raises(ValueError, match="Invalid username format"):
            Username("john-doe")  # hyphen not allowed
        
        with pytest.raises(ValueError, match="Invalid username format"):
            Username("john.doe")  # dot not allowed
        
        with pytest.raises(ValueError, match="Invalid username format"):
            Username("john doe")  # space not allowed
    
    def test_reserved_usernames(self):
        """Test reserved usernames cannot be used."""
        with pytest.raises(ValueError, match="is reserved"):
            Username("admin")
        
        with pytest.raises(ValueError, match="is reserved"):
            Username("root")
        
        with pytest.raises(ValueError, match="is reserved"):
            Username("system")
    
    def test_username_immutable(self):
        """Test username is immutable."""
        username = Username("johndoe")
        with pytest.raises(AttributeError):
            username.value = "different"  # type: ignore


class TestPassword:
    """Tests for Password value object."""
    
    def test_valid_password(self):
        """Test creating a valid password."""
        password = Password("password123")
        assert len(password.value) >= 8
    
    def test_password_minimum_length(self):
        """Test password must be at least 8 characters."""
        with pytest.raises(ValueError, match="must be at least 8 characters"):
            Password("pass123")
    
    def test_password_maximum_length(self):
        """Test password cannot exceed 128 characters."""
        long_password = "a" * 129
        with pytest.raises(ValueError, match="cannot exceed 128 characters"):
            Password(long_password)
    
    def test_password_must_contain_letter(self):
        """Test password must contain at least one letter."""
        with pytest.raises(ValueError, match="must contain at least one letter"):
            Password("12345678")
    
    def test_password_must_contain_number(self):
        """Test password must contain at least one number."""
        with pytest.raises(ValueError, match="must contain at least one number"):
            Password("passwordonly")
    
    def test_password_string_representation_masked(self):
        """Test password string representation is masked."""
        password = Password("password123")
        assert str(password) == "***********"
        assert "***" in repr(password)
        assert "password123" not in repr(password)
    
    def test_password_strength_weak(self):
        """Test weak password strength."""
        password = Password("pass1234")  # Short, no uppercase, no special chars
        assert password.strength == "weak"
    
    def test_password_strength_medium(self):
        """Test medium password strength."""
        password = Password("Password123")  # Good length, uppercase, lowercase, number
        assert password.strength == "medium"
    
    def test_password_strength_strong(self):
        """Test strong password strength."""
        password = Password("StrongPass123!")  # Long, uppercase, lowercase, number, special char
        assert password.strength == "strong"
    
    def test_password_immutable(self):
        """Test password is immutable."""
        password = Password("password123")
        with pytest.raises(AttributeError):
            password.value = "different123"  # type: ignore


class TestAccountStatus:
    """Tests for AccountStatus enumeration."""
    
    def test_account_statuses(self):
        """Test all account status values."""
        assert AccountStatus.ACTIVE.value == "active"
        assert AccountStatus.INACTIVE.value == "inactive"
        assert AccountStatus.SUSPENDED.value == "suspended"
        assert AccountStatus.BANNED.value == "banned"
        assert AccountStatus.PENDING_VERIFICATION.value == "pending_verification"


class TestPrivacyLevel:
    """Tests for PrivacyLevel enumeration."""
    
    def test_privacy_levels(self):
        """Test all privacy level values."""
        assert PrivacyLevel.PUBLIC.value == "public"
        assert PrivacyLevel.FRIENDS.value == "friends"
        assert PrivacyLevel.PRIVATE.value == "private"


class TestSuspensionDetails:
    """Tests for SuspensionDetails value object."""
    
    def test_valid_temporary_suspension(self):
        """Test creating temporary suspension details."""
        now = datetime.utcnow()
        ends_at = now + timedelta(days=7)
        
        suspension = SuspensionDetails(
            reason="Spam posting",
            suspended_at=now,
            ends_at=ends_at,
        )
        
        assert suspension.reason == "Spam posting"
        assert suspension.is_permanent is False
        assert suspension.is_expired is False
    
    def test_valid_permanent_suspension(self):
        """Test creating permanent suspension details."""
        now = datetime.utcnow()
        
        suspension = SuspensionDetails(
            reason="Repeated violations",
            suspended_at=now,
            ends_at=None,
        )
        
        assert suspension.is_permanent is True
        assert suspension.is_expired is False
    
    def test_expired_suspension(self):
        """Test expired suspension detection."""
        past = datetime.utcnow() - timedelta(days=10)
        past_end = datetime.utcnow() - timedelta(days=3)
        
        suspension = SuspensionDetails(
            reason="Temporary ban",
            suspended_at=past,
            ends_at=past_end,
        )
        
        assert suspension.is_expired is True
    
    def test_empty_reason_raises_error(self):
        """Test empty suspension reason raises ValueError."""
        with pytest.raises(ValueError, match="Suspension reason cannot be empty"):
            SuspensionDetails(
                reason="",
                suspended_at=datetime.utcnow(),
                ends_at=None,
            )
    
    def test_end_date_before_start_raises_error(self):
        """Test end date before start date raises ValueError."""
        now = datetime.utcnow()
        past = now - timedelta(days=1)
        
        with pytest.raises(ValueError, match="must be after start date"):
            SuspensionDetails(
                reason="Test",
                suspended_at=now,
                ends_at=past,
            )
    
    def test_suspension_immutable(self):
        """Test suspension details are immutable."""
        suspension = SuspensionDetails(
            reason="Test",
            suspended_at=datetime.utcnow(),
            ends_at=None,
        )
        
        with pytest.raises(AttributeError):
            suspension.reason = "Different reason"  # type: ignore


class TestBanDetails:
    """Tests for BanDetails value object."""
    
    def test_valid_ban(self):
        """Test creating valid ban details."""
        now = datetime.utcnow()
        ban = BanDetails(
            reason="Severe policy violation",
            banned_at=now,
        )
        
        assert ban.reason == "Severe policy violation"
        assert ban.banned_at == now
    
    def test_empty_reason_raises_error(self):
        """Test empty ban reason raises ValueError."""
        with pytest.raises(ValueError, match="Ban reason cannot be empty"):
            BanDetails(
                reason="",
                banned_at=datetime.utcnow(),
            )
    
    def test_ban_immutable(self):
        """Test ban details are immutable."""
        ban = BanDetails(
            reason="Test ban",
            banned_at=datetime.utcnow(),
        )
        
        with pytest.raises(AttributeError):
            ban.reason = "Different reason"  # type: ignore
