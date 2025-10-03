"""
Integration tests for authentication functionality.

This module contains integration tests for the authentication
endpoints and service integration.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestAuthIntegration:
    """Integration tests for authentication endpoints."""

    def test_register_user_success(self, client: TestClient):
        """Test successful user registration."""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPassword123",  # Valid password with upper, lower, and digit
            "display_name": "Test User",
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert "verification_token" in data

    def test_register_user_duplicate_email(self, client: TestClient, test_user):
        """Test user registration with duplicate email."""
        user_data = {
            "username": "newuser",
            "email": test_user.email,  # Duplicate email
            "password": "TestPassword123",
            "display_name": "New User",
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_register_user_invalid_data(self, client: TestClient):
        """Test user registration with invalid data."""
        user_data = {
            "username": "testuser",
            "email": "invalid-email",  # Invalid email
            "password": "123",  # Too short
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == 422

    def test_verify_email_success(self, client: TestClient):
        """Test successful email verification."""
        verification_data = {
            "token": "valid_verification_token",
        }
        
        response = client.post("/api/v1/auth/verify-email", json=verification_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "verified"

    def test_verify_email_invalid_token(self, client: TestClient):
        """Test email verification with invalid token."""
        verification_data = {
            "token": "invalid_token",
        }
        
        response = client.post("/api/v1/auth/verify-email", json=verification_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_login_success(self, client: TestClient, test_user):
        """Test successful login."""
        login_data = {
            "email": test_user.email,
            "password": "TestPassword123",  # Match the password from conftest
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["id"] == str(test_user.id)

    def test_login_invalid_credentials(self, client: TestClient):
        """Test login with invalid credentials."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword",
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_refresh_token_success(self, client: TestClient):
        """Test successful token refresh."""
        refresh_data = {
            "refresh_token": "valid_refresh_token",
        }
        
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_token_invalid(self, client: TestClient):
        """Test token refresh with invalid refresh token."""
        refresh_data = {
            "refresh_token": "invalid_refresh_token",
        }
        
        response = client.post("/api/v1/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_logout_success(self, client: TestClient, auth_headers):
        """Test successful logout."""
        logout_data = {
            "refresh_token": "valid_refresh_token",
        }
        
        response = client.post("/api/v1/auth/logout", json=logout_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_logout_unauthorized(self, client: TestClient):
        """Test logout without authentication."""
        logout_data = {
            "refresh_token": "valid_refresh_token",
        }
        
        response = client.post("/api/v1/auth/logout", json=logout_data)
        
        assert response.status_code == 401

    def test_reset_password_request_success(self, client: TestClient, test_user):
        """Test successful password reset request."""
        reset_data = {
            "email": test_user.email,
        }
        
        response = client.post("/api/v1/auth/reset-password-request", json=reset_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "reset_token" in data

    def test_reset_password_request_invalid_email(self, client: TestClient):
        """Test password reset request with invalid email."""
        reset_data = {
            "email": "nonexistent@example.com",
        }
        
        response = client.post("/api/v1/auth/reset-password-request", json=reset_data)
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_reset_password_success(self, client: TestClient):
        """Test successful password reset."""
        reset_data = {
            "token": "valid_reset_token",
            "new_password": "newpassword123",
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_reset_password_invalid_token(self, client: TestClient):
        """Test password reset with invalid token."""
        reset_data = {
            "token": "invalid_reset_token",
            "new_password": "newpassword123",
        }
        
        response = client.post("/api/v1/auth/reset-password", json=reset_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_change_password_success(self, client: TestClient, auth_headers):
        """Test successful password change."""
        password_data = {
            "current_password": "currentpassword123",
            "new_password": "newpassword123",
        }
        
        response = client.put("/api/v1/auth/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_change_password_unauthorized(self, client: TestClient):
        """Test password change without authentication."""
        password_data = {
            "current_password": "currentpassword123",
            "new_password": "newpassword123",
        }
        
        response = client.put("/api/v1/auth/change-password", json=password_data)
        
        assert response.status_code == 401

    def test_enable_two_factor_success(self, client: TestClient, auth_headers):
        """Test successful two-factor authentication enablement."""
        response = client.post("/api/v1/auth/enable-2fa", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "qr_code" in data
        assert "secret" in data

    def test_enable_two_factor_unauthorized(self, client: TestClient):
        """Test two-factor authentication enablement without authentication."""
        response = client.post("/api/v1/auth/enable-2fa")
        
        assert response.status_code == 401

    def test_verify_two_factor_success(self, client: TestClient, auth_headers):
        """Test successful two-factor authentication verification."""
        verification_data = {
            "token": "123456",
        }
        
        response = client.post("/api/v1/auth/verify-2fa", json=verification_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "verified"

    def test_disable_two_factor_success(self, client: TestClient, auth_headers):
        """Test successful two-factor authentication disablement."""
        response = client.post("/api/v1/auth/disable-2fa", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disabled"

    def test_social_login_success(self, client: TestClient):
        """Test successful social login."""
        social_data = {
            "provider": "google",
            "provider_id": "google_123",
            "email": "test@example.com",
            "name": "Test User",
        }
        
        response = client.post("/api/v1/auth/social-login", json=social_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_get_user_profile_success(self, client: TestClient, auth_headers):
        """Test successful user profile retrieval."""
        response = client.get("/api/v1/auth/profile", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data

    def test_get_user_profile_unauthorized(self, client: TestClient):
        """Test user profile retrieval without authentication."""
        response = client.get("/api/v1/auth/profile")
        
        assert response.status_code == 401

    def test_update_user_preferences_success(self, client: TestClient, auth_headers):
        """Test successful user preferences update."""
        preferences = {
            "notifications": {"email": True, "push": False},
            "privacy": {"profile_visibility": "public"},
        }
        
        response = client.put("/api/v1/auth/preferences", json=preferences, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_update_user_preferences_unauthorized(self, client: TestClient):
        """Test user preferences update without authentication."""
        preferences = {
            "notifications": {"email": True, "push": False},
        }
        
        response = client.put("/api/v1/auth/preferences", json=preferences)
        
        assert response.status_code == 401
