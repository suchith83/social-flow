"""MFA (Multi-Factor Authentication) Tests."""

import pytest
import pyotp
from datetime import datetime
from unittest.mock import Mock, patch


class TestAuthMFA:
    """Test MFA/2FA authentication functionality."""

    def test_mfa_secret_generation(self):
        """Test MFA secret key generation."""
        secret = pyotp.random_base32()
        assert secret is not None
        assert len(secret) == 32
        assert secret.isalnum()

    def test_totp_generation(self):
        """Test TOTP (Time-based OTP) generation."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        code = totp.now()
        assert code is not None
        assert len(code) == 6
        assert code.isdigit()

    def test_totp_validation_success(self):
        """Test successful TOTP validation."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        code = totp.now()
        assert totp.verify(code) is True

    def test_totp_validation_failure(self):
        """Test TOTP validation with wrong code."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        assert totp.verify("000000") is False

    def test_mfa_qr_code_generation(self):
        """Test QR code URI generation for MFA setup."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(
            name="testuser@example.com",
            issuer_name="SocialFlow"
        )
        assert "otpauth://totp/" in uri
        assert "SocialFlow" in uri
        assert "secret=" in uri

    def test_backup_codes_generation(self):
        """Test backup codes generation."""
        import secrets
        backup_codes = [secrets.token_hex(4) for _ in range(10)]
        assert len(backup_codes) == 10
        assert all(len(code) == 8 for code in backup_codes)
        assert len(set(backup_codes)) == 10  # All unique

    @pytest.mark.parametrize("interval", [30, 60])
    def test_totp_time_window(self, interval):
        """Test TOTP with different time intervals."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret, interval=interval)
        code = totp.now()
        assert len(code) == 6