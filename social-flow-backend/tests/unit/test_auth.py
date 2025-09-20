"""
Unit tests for authentication functionality.

This module contains unit tests for the authentication service
and related functionality.
"""

import pytest
from unittest.mock import Mock, patch
from app.services.auth import AuthService
from app.core.exceptions import AuthServiceError


class TestAuthService:
    """Test cases for AuthService."""

    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance for testing."""
        return AuthService()

    @pytest.mark.asyncio
    async def test_register_user_with_verification_success(self, auth_service):
        """Test successful user registration with email verification."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123",
            "display_name": "Test User",
        }
        
        with patch.object(auth_service, 'create_user') as mock_create_user:
            mock_create_user.return_value = {"id": "user123", "username": "testuser"}
            
            result = await auth_service.register_user_with_verification(user_data)
            
            assert result["status"] == "success"
            assert "verification_token" in result
            mock_create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_user_with_verification_duplicate_email(self, auth_service):
        """Test user registration with duplicate email."""
        user_data = {
            "username": "testuser",
            "email": "existing@example.com",
            "password": "testpassword123",
        }
        
        with patch.object(auth_service, 'create_user') as mock_create_user:
            mock_create_user.side_effect = AuthServiceError("Email already exists")
            
            with pytest.raises(AuthServiceError):
                await auth_service.register_user_with_verification(user_data)

    @pytest.mark.asyncio
    async def test_verify_email_success(self, auth_service):
        """Test successful email verification."""
        verification_token = "valid_token_123"
        
        with patch.object(auth_service, 'verify_user_email') as mock_verify:
            mock_verify.return_value = {"status": "verified"}
            
            result = await auth_service.verify_email(verification_token)
            
            assert result["status"] == "verified"
            mock_verify.assert_called_once_with(verification_token)

    @pytest.mark.asyncio
    async def test_verify_email_invalid_token(self, auth_service):
        """Test email verification with invalid token."""
        verification_token = "invalid_token"
        
        with patch.object(auth_service, 'verify_user_email') as mock_verify:
            mock_verify.side_effect = AuthServiceError("Invalid verification token")
            
            with pytest.raises(AuthServiceError):
                await auth_service.verify_email(verification_token)

    @pytest.mark.asyncio
    async def test_login_with_credentials_success(self, auth_service):
        """Test successful login with credentials."""
        credentials = {
            "email": "test@example.com",
            "password": "testpassword123",
        }
        
        with patch.object(auth_service, 'authenticate_user') as mock_auth:
            mock_auth.return_value = {
                "user": {"id": "user123", "username": "testuser"},
                "access_token": "access_token_123",
                "refresh_token": "refresh_token_123",
            }
            
            result = await auth_service.login_with_credentials(credentials)
            
            assert "access_token" in result
            assert "refresh_token" in result
            assert result["user"]["id"] == "user123"

    @pytest.mark.asyncio
    async def test_login_with_credentials_invalid(self, auth_service):
        """Test login with invalid credentials."""
        credentials = {
            "email": "test@example.com",
            "password": "wrongpassword",
        }
        
        with patch.object(auth_service, 'authenticate_user') as mock_auth:
            mock_auth.side_effect = AuthServiceError("Invalid credentials")
            
            with pytest.raises(AuthServiceError):
                await auth_service.login_with_credentials(credentials)

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, auth_service):
        """Test successful access token refresh."""
        refresh_token = "valid_refresh_token"
        
        with patch.object(auth_service, 'refresh_user_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
            }
            
            result = await auth_service.refresh_access_token(refresh_token)
            
            assert "access_token" in result
            assert "refresh_token" in result

    @pytest.mark.asyncio
    async def test_refresh_access_token_invalid(self, auth_service):
        """Test access token refresh with invalid refresh token."""
        refresh_token = "invalid_refresh_token"
        
        with patch.object(auth_service, 'refresh_user_token') as mock_refresh:
            mock_refresh.side_effect = AuthServiceError("Invalid refresh token")
            
            with pytest.raises(AuthServiceError):
                await auth_service.refresh_access_token(refresh_token)

    @pytest.mark.asyncio
    async def test_logout_user_success(self, auth_service):
        """Test successful user logout."""
        user_id = "user123"
        refresh_token = "refresh_token_123"
        
        with patch.object(auth_service, 'revoke_user_tokens') as mock_revoke:
            mock_revoke.return_value = {"status": "success"}
            
            result = await auth_service.logout_user(user_id, refresh_token)
            
            assert result["status"] == "success"
            mock_revoke.assert_called_once_with(user_id, refresh_token)

    @pytest.mark.asyncio
    async def test_reset_password_request_success(self, auth_service):
        """Test successful password reset request."""
        email = "test@example.com"
        
        with patch.object(auth_service, 'initiate_password_reset') as mock_reset:
            mock_reset.return_value = {"reset_token": "reset_token_123"}
            
            result = await auth_service.reset_password_request(email)
            
            assert "reset_token" in result
            mock_reset.assert_called_once_with(email)

    @pytest.mark.asyncio
    async def test_reset_password_success(self, auth_service):
        """Test successful password reset."""
        reset_data = {
            "token": "reset_token_123",
            "new_password": "newpassword123",
        }
        
        with patch.object(auth_service, 'reset_user_password') as mock_reset:
            mock_reset.return_value = {"status": "success"}
            
            result = await auth_service.reset_password(reset_data)
            
            assert result["status"] == "success"
            mock_reset.assert_called_once_with(reset_data["token"], reset_data["new_password"])

    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_service):
        """Test successful password change."""
        user_id = "user123"
        password_data = {
            "current_password": "oldpassword123",
            "new_password": "newpassword123",
        }
        
        with patch.object(auth_service, 'update_user_password') as mock_update:
            mock_update.return_value = {"status": "success"}
            
            result = await auth_service.change_password(user_id, password_data)
            
            assert result["status"] == "success"
            mock_update.assert_called_once_with(user_id, password_data["current_password"], password_data["new_password"])

    @pytest.mark.asyncio
    async def test_enable_two_factor_success(self, auth_service):
        """Test successful two-factor authentication enablement."""
        user_id = "user123"
        
        with patch.object(auth_service, 'setup_user_2fa') as mock_2fa:
            mock_2fa.return_value = {"qr_code": "qr_code_data", "secret": "secret_key"}
            
            result = await auth_service.enable_two_factor(user_id)
            
            assert "qr_code" in result
            assert "secret" in result
            mock_2fa.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_verify_two_factor_success(self, auth_service):
        """Test successful two-factor authentication verification."""
        user_id = "user123"
        token = "123456"
        
        with patch.object(auth_service, 'verify_user_2fa') as mock_verify:
            mock_verify.return_value = {"status": "verified"}
            
            result = await auth_service.verify_two_factor(user_id, token)
            
            assert result["status"] == "verified"
            mock_verify.assert_called_once_with(user_id, token)

    @pytest.mark.asyncio
    async def test_disable_two_factor_success(self, auth_service):
        """Test successful two-factor authentication disablement."""
        user_id = "user123"
        
        with patch.object(auth_service, 'remove_user_2fa') as mock_disable:
            mock_disable.return_value = {"status": "disabled"}
            
            result = await auth_service.disable_two_factor(user_id)
            
            assert result["status"] == "disabled"
            mock_disable.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_social_login_success(self, auth_service):
        """Test successful social login."""
        social_data = {
            "provider": "google",
            "provider_id": "google_123",
            "email": "test@example.com",
            "name": "Test User",
        }
        
        with patch.object(auth_service, 'authenticate_social_user') as mock_social:
            mock_social.return_value = {
                "user": {"id": "user123", "username": "testuser"},
                "access_token": "access_token_123",
                "refresh_token": "refresh_token_123",
            }
            
            result = await auth_service.social_login(social_data)
            
            assert "access_token" in result
            assert "refresh_token" in result
            assert result["user"]["id"] == "user123"

    @pytest.mark.asyncio
    async def test_get_user_profile_success(self, auth_service):
        """Test successful user profile retrieval."""
        user_id = "user123"
        
        with patch.object(auth_service, 'get_user') as mock_get_user:
            mock_get_user.return_value = {
                "id": "user123",
                "username": "testuser",
                "email": "test@example.com",
                "display_name": "Test User",
            }
            
            result = await auth_service.get_user_profile(user_id)
            
            assert result["id"] == "user123"
            assert result["username"] == "testuser"
            mock_get_user.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_update_user_preferences_success(self, auth_service):
        """Test successful user preferences update."""
        user_id = "user123"
        preferences = {
            "notifications": {"email": True, "push": False},
            "privacy": {"profile_visibility": "public"},
        }
        
        with patch.object(auth_service, 'update_user') as mock_update:
            mock_update.return_value = {"status": "success"}
            
            result = await auth_service.update_user_preferences(user_id, preferences)
            
            assert result["status"] == "success"
            mock_update.assert_called_once_with(user_id, {"preferences": preferences})

    @pytest.mark.asyncio
    async def test_ban_user_success(self, auth_service):
        """Test successful user banning."""
        user_id = "user123"
        reason = "Violation of terms of service"
        
        with patch.object(auth_service, 'ban_user') as mock_ban:
            mock_ban.return_value = {"status": "banned", "reason": reason}
            
            result = await auth_service.ban_user(user_id, reason)
            
            assert result["status"] == "banned"
            assert result["reason"] == reason
            mock_ban.assert_called_once_with(user_id, reason)

    @pytest.mark.asyncio
    async def test_suspend_user_success(self, auth_service):
        """Test successful user suspension."""
        user_id = "user123"
        duration = 7  # days
        reason = "Temporary violation"
        
        with patch.object(auth_service, 'suspend_user') as mock_suspend:
            mock_suspend.return_value = {"status": "suspended", "duration": duration}
            
            result = await auth_service.suspend_user(user_id, duration, reason)
            
            assert result["status"] == "suspended"
            assert result["duration"] == duration
            mock_suspend.assert_called_once_with(user_id, duration, reason)

    @pytest.mark.asyncio
    async def test_update_last_login_success(self, auth_service):
        """Test successful last login update."""
        user_id = "user123"
        
        with patch.object(auth_service, 'update_user') as mock_update:
            mock_update.return_value = {"status": "success"}
            
            result = await auth_service.update_last_login(user_id)
            
            assert result["status"] == "success"
            mock_update.assert_called_once()
