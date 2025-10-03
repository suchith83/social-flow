"""
Integration tests for authentication endpoints.

Tests all authentication flows including registration, login, 2FA, and token management.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_password
from app.infrastructure.crud import user as crud_user
from app.models.user import UserRole, UserStatus


class TestUserRegistration:
    """Test user registration endpoint."""
    
    @pytest.mark.asyncio
    async def test_register_success(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
    ):
        """Test successful user registration."""
        # Arrange
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "ValidPassword123!",
            "display_name": "New User",
        }
        
        # Act
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert data["display_name"] == user_data["display_name"]
        assert "id" in data
        assert "password" not in data
        assert "password_hash" not in data
        
        # Verify user exists in database
        user = await crud_user.get_by_email(db_session, email=user_data["email"])
        assert user is not None
        assert user.email == user_data["email"]
        assert user.username == user_data["username"]
        assert verify_password(user_data["password"], user.password_hash)
        assert user.role == UserRole.USER
        assert user.status == UserStatus.PENDING_VERIFICATION  # New users start as pending
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test registration with duplicate email fails."""
        # Arrange
        user_data = {
            "email": test_user.email,  # Duplicate email
            "username": "differentuser",
            "password": "ValidPassword123!",
            "display_name": "Different User",
        }
        
        # Act
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "email" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_register_duplicate_username(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test registration with duplicate username fails."""
        # Arrange
        user_data = {
            "email": "different@example.com",
            "username": test_user.username,  # Duplicate username
            "password": "ValidPassword123!",
            "display_name": "Different User",
        }
        
        # Act
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "username" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_register_invalid_email(
        self,
        async_client: AsyncClient,
    ):
        """Test registration with invalid email fails."""
        # Arrange
        user_data = {
            "email": "not-an-email",
            "username": "validuser",
            "password": "ValidPassword123!",
            "display_name": "Valid User",
        }
        
        # Act
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_register_weak_password(
        self,
        async_client: AsyncClient,
    ):
        """Test registration with weak password fails."""
        # Arrange
        user_data = {
            "email": "user@example.com",
            "username": "validuser",
            "password": "weak",  # Too short, no numbers/special chars
            "display_name": "Valid User",
        }
        
        # Act
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 422  # Validation error


class TestUserLogin:
    """Test user login endpoints."""
    
    @pytest.mark.asyncio
    async def test_oauth2_login_success(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test successful OAuth2 login."""
        # Arrange
        login_data = {
            "username": test_user.email,
            "password": "TestPassword123",
        }
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,  # OAuth2 uses form data
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    @pytest.mark.asyncio
    async def test_json_login_success(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test successful JSON login."""
        # Arrange
        login_data = {
            "username_or_email": test_user.email,
            "password": "TestPassword123",
        }
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/login/json",
            json=login_data,  # JSON format
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_with_username(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test login with username instead of email."""
        # Arrange
        login_data = {
            "username": test_user.username,
            "password": "TestPassword123",
        }
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test login with wrong password fails."""
        # Arrange
        login_data = {
            "username": test_user.email,
            "password": "WrongPassword123",
        }
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )
        
        # Assert
        assert response.status_code == 401
        data = response.json()
        assert "incorrect" in data["detail"].lower() or "invalid" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_login_nonexistent_user(
        self,
        async_client: AsyncClient,
    ):
        """Test login with non-existent user fails."""
        # Arrange
        login_data = {
            "username": "nonexistent@example.com",
            "password": "SomePassword123",
        }
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/login",
            data=login_data,
        )
        
        # Assert
        assert response.status_code == 401


class TestTokenRefresh:
    """Test token refresh endpoint."""
    
    @pytest.mark.asyncio
    async def test_token_refresh_success(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test successful token refresh."""
        # Step 1: Login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Step 2: Refresh access token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_token_refresh_invalid_token(
        self,
        async_client: AsyncClient,
    ):
        """Test token refresh with invalid token fails."""
        # Arrange
        invalid_token = "invalid.token.here"
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": invalid_token},
        )
        
        # Assert
        assert response.status_code == 401


class TestTwoFactorAuth:
    """Test two-factor authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_2fa_setup(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test 2FA setup returns secret and QR code."""
        # Arrange - Get auth token
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/2fa/setup",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "secret" in data
        assert "qr_code_url" in data
        assert "backup_codes" in data
        assert len(data["secret"]) > 0
        assert "otpauth://" in data["qr_code_url"]
        assert len(data["backup_codes"]) == 10
    
    @pytest.mark.asyncio
    async def test_2fa_setup_unauthenticated(
        self,
        async_client: AsyncClient,
    ):
        """Test 2FA setup without authentication fails."""
        # Act
        response = await async_client.post("/api/v1/auth/2fa/setup")
        
        # Assert
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_2fa_disable(
        self,
        async_client: AsyncClient,
        test_user,
        db_session: AsyncSession,
    ):
        """Test disabling 2FA."""
        # Arrange - Enable 2FA first
        test_user.two_factor_secret = "TESTSECRET123456"
        test_user.two_factor_enabled = True
        db_session.add(test_user)
        await db_session.commit()
        
        # Get auth token
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            "/api/v1/auth/2fa/disable",
            headers=headers,
            json={"password": "TestPassword123"},
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "2FA disabled successfully" or "disabled" in data["message"].lower()
        assert data["success"] is True


class TestCurrentUser:
    """Test get current user endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test getting current authenticated user."""
        # Arrange - Get auth token
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_user.id)
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username
        assert "password" not in data
        assert "password_hash" not in data
    
    @pytest.mark.asyncio
    async def test_get_current_user_unauthenticated(
        self,
        async_client: AsyncClient,
    ):
        """Test getting current user without authentication fails."""
        # Act
        response = await async_client.get("/api/v1/auth/me")
        
        # Assert
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(
        self,
        async_client: AsyncClient,
    ):
        """Test getting current user with invalid token fails."""
        # Arrange
        headers = {"Authorization": "Bearer invalid.token.here"}
        
        # Act
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 401


class TestAuthEdgeCases:
    """Test edge cases and security scenarios."""
    
    @pytest.mark.asyncio
    async def test_register_with_extra_fields(
        self,
        async_client: AsyncClient,
    ):
        """Test that extra fields in registration are ignored."""
        # Arrange
        user_data = {
            "email": "user@example.com",
            "username": "testuser",
            "password": "ValidPassword123!",
            "display_name": "Test User",
            "role": "admin",  # Should be ignored
            "is_superuser": True,  # Should be ignored
        }
        
        # Act
        response = await async_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        # Should have default USER role, not admin
        assert data.get("role", "user") == "user"
    
    @pytest.mark.asyncio
    async def test_login_updates_last_login(
        self,
        async_client: AsyncClient,
        test_user,
        db_session: AsyncSession,
    ):
        """Test that login updates last_login timestamp."""
        # Arrange
        original_last_login = test_user.last_login_at
        
        # Act
        await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        
        # Assert
        await db_session.refresh(test_user)
        assert test_user.last_login_at != original_last_login
        assert test_user.last_login_at is not None
    
    @pytest.mark.asyncio
    async def test_multiple_logins_different_tokens(
        self,
        async_client: AsyncClient,
        test_user,
    ):
        """Test that multiple logins generate different tokens."""
        # Act - Login twice
        response1 = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        response2 = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        
        # Assert
        token1 = response1.json()["access_token"]
        token2 = response2.json()["access_token"]
        assert token1 != token2
