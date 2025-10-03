"""
Integration tests for User API endpoints (Clean Architecture).

Tests the complete flow from HTTP request through application services
to domain entities and repositories.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient


class TestUserRegistration:
    """Test user registration endpoint."""
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, async_client: AsyncClient):
        """Test successful user registration."""
        response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "Test123!",  # Shortened to avoid bcrypt 72-byte limit issues
                "display_name": "Test User",
            },
        )
        
        # Debug: print response if not 201
        if response.status_code != 201:
            print(f"\nUnexpected status code: {response.status_code}")
            print(f"Response body: {response.text}")
            try:
                print(f"Response JSON: {response.json()}")
            except Exception:
                pass
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["display_name"] == "Test User"
        assert not data["is_verified"]
        assert data["is_active"]
        assert "id" in data
    
    @pytest.mark.asyncio
    async def test_register_user_duplicate_username(self, async_client: AsyncClient):
        """Test registration with duplicate username."""
        # First registration
        await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test1@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User 1",
            },
        )
        
        # Second registration with same username
        response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test2@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User 2",
            },
        )
        
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(self, async_client: AsyncClient):
        """Test registration with duplicate email."""
        # First registration
        await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser1",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User 1",
            },
        )
        
        # Second registration with same email
        response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser2",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User 2",
            },
        )
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_register_user_invalid_email(self, async_client: AsyncClient):
        """Test registration with invalid email."""
        response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "invalid-email",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_register_user_short_password(self, async_client: AsyncClient):
        """Test registration with too short password."""
        response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "short",
                "display_name": "Test User",
            },
        )
        
        assert response.status_code == 422  # Validation error


class TestUserAuthentication:
    """Test user authentication endpoint."""
    
    @pytest.mark.asyncio
    async def test_login_success_with_username(self, async_client: AsyncClient):
        """Test successful login with username."""
        # Register user
        await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        
        # Login
        response = await async_client.post(
            "/api/v1/v2/users/login",
            json={
                "username_or_email": "testuser",
                "password": "SecurePass123!",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "user" in data
        assert data["user"]["username"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_login_success_with_email(self, async_client: AsyncClient):
        """Test successful login with email."""
        # Register user
        await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        
        # Login
        response = await async_client.post(
            "/api/v1/v2/users/login",
            json={
                "username_or_email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, async_client: AsyncClient):
        """Test login with wrong password."""
        # Register user
        await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        
        # Login with wrong password
        response = await async_client.post(
            "/api/v1/v2/users/login",
            json={
                "username_or_email": "testuser",
                "password": "WrongPassword123!",
            },
        )
        
        assert response.status_code == 401
        assert "invalid credentials" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, async_client: AsyncClient):
        """Test login with nonexistent user."""
        response = await async_client.post(
            "/api/v1/v2/users/login",
            json={
                "username_or_email": "nonexistent",
                "password": "SecurePass123!",
            },
        )
        
        assert response.status_code == 401


class TestUserProfile:
    """Test user profile endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, async_client: AsyncClient):
        """Test getting user by ID."""
        # Register user
        register_response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        user_id = register_response.json()["id"]
        
        # Get user by ID
        response = await async_client.get(f"/api/v1/v2/users/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_get_user_by_username(self, async_client: AsyncClient):
        """Test getting user by username."""
        # Register user
        await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        
        # Get user by username
        response = await async_client.get("/api/v1/v2/users/username/testuser")
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, async_client: AsyncClient):
        """Test getting nonexistent user."""
        response = await async_client.get(
            "/api/v1/v2/users/00000000-0000-0000-0000-000000000000"
        )
        
        assert response.status_code == 404


class TestUserSearch:
    """Test user search endpoint."""
    
    @pytest.mark.asyncio
    async def test_search_users(self, async_client: AsyncClient):
        """Test searching users."""
        # Register multiple users
        for i in range(3):
            await async_client.post(
                "/api/v1/v2/users/register",
                json={
                    "username": f"testuser{i}",
                    "email": f"test{i}@example.com",
                    "password": "SecurePass123!",
                    "display_name": f"Test User {i}",
                },
            )
        
        # Search users
        response = await async_client.get("/api/v1/v2/users/?query=testuser")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert all("testuser" in user["username"] for user in data)
    
    @pytest.mark.asyncio
    async def test_search_users_pagination(self, async_client: AsyncClient):
        """Test user search with pagination."""
        # Register multiple users
        for i in range(5):
            await async_client.post(
                "/api/v1/v2/users/register",
                json={
                    "username": f"testuser{i}",
                    "email": f"test{i}@example.com",
                    "password": "SecurePass123!",
                    "display_name": f"Test User {i}",
                },
            )
        
        # Search with pagination
        response = await async_client.get(
            "/api/v1/v2/users/?query=testuser&skip=2&limit=2"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestUserStatistics:
    """Test user statistics endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_followers(self, async_client: AsyncClient):
        """Test getting user followers."""
        # Register user
        register_response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        user_id = register_response.json()["id"]
        
        # Get followers (should be empty initially)
        response = await async_client.get(f"/api/v1/v2/users/{user_id}/followers")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_following(self, async_client: AsyncClient):
        """Test getting users that user follows."""
        # Register user
        register_response = await async_client.post(
            "/api/v1/v2/users/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePass123!",
                "display_name": "Test User",
            },
        )
        user_id = register_response.json()["id"]
        
        # Get following (should be empty initially)
        response = await async_client.get(f"/api/v1/v2/users/{user_id}/following")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
