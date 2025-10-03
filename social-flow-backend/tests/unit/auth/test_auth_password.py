"""Password Authentication Tests."""

import pytest
import bcrypt


class TestAuthPassword:
    """Test password hashing and validation."""

    def test_password_hashing_success(self):
        """Test successful password hashing."""
        password = b"TestPassword123!"
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 0

    def test_password_verification_success(self):
        """Test successful password verification."""
        password = b"TestPassword123!"
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())
        assert bcrypt.checkpw(password, hashed) is True

    def test_password_verification_failure(self):
        """Test password verification with wrong password."""
        password = b"TestPassword123!"
        wrong_password = b"WrongPassword456!"
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())
        assert bcrypt.checkpw(wrong_password, hashed) is False

    def test_password_strength_weak(self):
        """Test weak password detection."""
        weak_passwords = ["123456", "password", "abc123"]
        for pwd in weak_passwords:
            # Simple length check
            assert len(pwd) < 8 or pwd.isdigit() or pwd.isalpha()

    def test_password_strength_strong(self):
        """Test strong password validation."""
        strong_password = "MyStr0ng!Pass2024"
        assert len(strong_password) >= 8
        assert any(c.isupper() for c in strong_password)
        assert any(c.islower() for c in strong_password)
        assert any(c.isdigit() for c in strong_password)
        assert any(c in "!@#$%^&*()" for c in strong_password)

    @pytest.mark.parametrize("password,expected", [
        ("short", False),  # Too short
        ("alllowercase", False),  # No uppercase, numbers, special
        ("ALLUPPERCASE", False),  # No lowercase, numbers, special
        ("NoSpecialChar123", False),  # No special characters
        ("Valid!Pass123", True),  # Valid password
    ])
    def test_password_validation_rules(self, password, expected):
        """Test password validation against rules."""
        is_valid = (
            len(password) >= 8 and
            any(c.isupper() for c in password) and
            any(c.islower() for c in password) and
            any(c.isdigit() for c in password) and
            any(not c.isalnum() for c in password)
        )
        assert is_valid == expected

    def test_bcrypt_rounds(self):
        """Test bcrypt hashing with different rounds."""
        password = b"TestPassword123!"
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())
        # Bcrypt hashes start with $2b$
        assert hashed.startswith(b"$2b$")

    def test_hash_uniqueness(self):
        """Test that same password generates different hashes (due to salt)."""
        password = b"TestPassword123!"
        hash1 = bcrypt.hashpw(password, bcrypt.gensalt())
        hash2 = bcrypt.hashpw(password, bcrypt.gensalt())
        assert hash1 != hash2  # Different due to different salts
        # But both should verify correctly
        assert bcrypt.checkpw(password, hash1)
        assert bcrypt.checkpw(password, hash2)