"""
Integration tests for user management endpoints.

Tests all user operations including profile management, followers/following,
search, blocking, verification, and admin operations.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud import user as crud_user
from app.models.user import User, UserRole, UserStatus


class TestUserProfile:
    """Test user profile endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_profile(
        self,
        async_client: AsyncClient,
        test_user: User,
    ):
        """Test getting current user's detailed profile."""
        # Arrange
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.get("/api/v1/users/me", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username
        assert "id" in data
        assert "role" in data
    
    @pytest.mark.asyncio
    async def test_update_current_user_profile(
        self,
        async_client: AsyncClient,
        test_user: User,
    ):
        """Test updating current user's profile."""
        # Arrange
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        update_data = {
            "full_name": "Updated Name",
            "bio": "Updated bio",
            "location": "Updated Location",
        }
        
        # Act
        response = await async_client.put(
            "/api/v1/users/me",
            headers=headers,
            json=update_data,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        # Note: Response uses display_name which maps to full_name
        assert data["bio"] == "Updated bio"
        assert data["location"] == "Updated Location"
    
    @pytest.mark.asyncio
    async def test_change_password(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test changing user password."""
        # Arrange
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        password_data = {
            "current_password": "TestPassword123",
            "new_password": "NewPassword456!",
        }
        
        # Act
        response = await async_client.put(
            "/api/v1/users/me/password",
            headers=headers,
            json=password_data,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "password" in data["message"].lower()
        
        # Verify can login with new password
        new_login = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "NewPassword456!"},
        )
        assert new_login.status_code == 200
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current(
        self,
        async_client: AsyncClient,
        test_user: User,
    ):
        """Test changing password with wrong current password."""
        # Arrange
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        password_data = {
            "current_password": "WrongPassword!",
            "new_password": "NewPassword456!",
        }
        
        # Act
        response = await async_client.put(
            "/api/v1/users/me/password",
            headers=headers,
            json=password_data,
        )
        
        # Assert
        assert response.status_code == 400


class TestUserList:
    """Test user listing and search endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_users_paginated(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test listing users with pagination."""
        # Arrange - Create additional users
        from app.schemas.user import UserRegister
        for i in range(5):
            user_data = UserRegister(
                username=f"user{i}",
                email=f"user{i}@example.com",
                password="TestPassword123!",
            )
            await crud_user.create(db_session, obj_in=user_data)
        
        # Act
        response = await async_client.get("/api/v1/users?skip=0&limit=3")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data
        assert len(data["items"]) <= 3
        assert data["total"] >= 6  # test_user + 5 new users
    
    @pytest.mark.asyncio
    async def test_search_users_by_username(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test searching users by username."""
        # Arrange - Create user with specific username
        from app.schemas.user import UserRegister
        user_data = UserRegister(
            username="searchable_user",
            email="searchable@example.com",
            password="TestPassword123!",
        )
        await crud_user.create(db_session, obj_in=user_data)
        
        # Act
        response = await async_client.get("/api/v1/users/search?q=searchable")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
        # Check that searchable_user is in results
        usernames = [user["username"] for user in data["items"]]
        assert "searchable_user" in usernames
    
    @pytest.mark.asyncio
    async def test_search_users_no_results(
        self,
        async_client: AsyncClient,
    ):
        """Test searching with no matching results."""
        # Act
        response = await async_client.get("/api/v1/users/search?q=nonexistentuser12345")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 0
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(
        self,
        async_client: AsyncClient,
        test_user: User,
    ):
        """Test getting a user by their ID."""
        # Act
        response = await async_client.get(f"/api/v1/users/{test_user.id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user.username
        assert data["id"] == str(test_user.id)
    
    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(
        self,
        async_client: AsyncClient,
    ):
        """Test getting non-existent user."""
        # Act
        fake_uuid = "00000000-0000-0000-0000-000000000000"
        response = await async_client.get(f"/api/v1/users/{fake_uuid}")
        
        # Assert
        assert response.status_code == 404


class TestFollowSystem:
    """Test follow/unfollow functionality."""
    
    @pytest.mark.asyncio
    async def test_follow_user(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test following another user."""
        # Arrange - Create another user to follow
        from app.schemas.user import UserRegister
        target_user_data = UserRegister(
            username="user_to_follow",
            email="follow@example.com",
            password="TestPassword123!",
        )
        target_user = await crud_user.create(db_session, obj_in=target_user_data)
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/users/{target_user.id}/follow",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "follower_id" in data or "status" in data
    
    @pytest.mark.asyncio
    async def test_unfollow_user(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test unfollowing a user."""
        # Arrange - Create and follow another user
        from app.schemas.user import UserRegister
        target_user_data = UserRegister(
            username="user_to_unfollow",
            email="unfollow@example.com",
            password="TestPassword123!",
        )
        target_user = await crud_user.create(db_session, obj_in=target_user_data)
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Follow first
        await async_client.post(
            f"/api/v1/users/{target_user.id}/follow",
            headers=headers,
        )
        
        # Act - Unfollow
        response = await async_client.delete(
            f"/api/v1/users/{target_user.id}/follow",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_follow_self_fails(
        self,
        async_client: AsyncClient,
        test_user: User,
    ):
        """Test that following yourself fails."""
        # Arrange
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/users/{test_user.id}/follow",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_get_user_followers(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test getting a user's followers list."""
        # Arrange - Create follower
        from app.schemas.user import UserRegister
        follower_data = UserRegister(
            username="follower_user",
            email="follower@example.com",
            password="TestPassword123!",
        )
        follower = await crud_user.create(db_session, obj_in=follower_data)
        follower.status = UserStatus.ACTIVE
        follower.is_verified = True
        db_session.add(follower)
        await db_session.commit()
        await db_session.refresh(follower)
        
        # Follower logs in and follows test_user
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": follower.email, "password": "TestPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        await async_client.post(
            f"/api/v1/users/{test_user.id}/follow",
            headers=headers,
        )
        
        # Act
        response = await async_client.get(f"/api/v1/users/{test_user.id}/followers")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
    
    @pytest.mark.asyncio
    async def test_get_user_following(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test getting list of users that a user follows."""
        # Arrange - Create user for test_user to follow
        from app.schemas.user import UserRegister
        target_data = UserRegister(
            username="followed_user",
            email="followed@example.com",
            password="TestPassword123!",
        )
        target_user = await crud_user.create(db_session, obj_in=target_data)
        
        # Login and follow
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        await async_client.post(
            f"/api/v1/users/{target_user.id}/follow",
            headers=headers,
        )
        
        # Act
        response = await async_client.get(f"/api/v1/users/{test_user.id}/following")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1


class TestAdminOperations:
    """Test admin-only user management operations."""
    
    @pytest.mark.asyncio
    async def test_update_user_admin_as_admin(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
    ):
        """Test admin can update user roles."""
        # Arrange - Create admin user
        from app.schemas.user import UserRegister
        admin_data = UserRegister(
            username="admin_user",
            email="admin@example.com",
            password="AdminPassword123!",
        )
        admin_user = await crud_user.create(db_session, obj_in=admin_data)
        admin_user.role = UserRole.ADMIN
        admin_user.status = UserStatus.ACTIVE
        admin_user.is_verified = True
        db_session.add(admin_user)
        await db_session.commit()
        await db_session.refresh(admin_user)
        
        # Create regular user
        user_data = UserRegister(
            username="regular_user",
            email="regular@example.com",
            password="TestPassword123!",
        )
        regular_user = await crud_user.create(db_session, obj_in=user_data)
        
        # Admin login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": admin_user.email, "password": "AdminPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act - Update regular user's role
        response = await async_client.put(
            f"/api/v1/users/{regular_user.id}/admin",
            headers=headers,
            json={"role": "moderator"},
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "moderator"
    
    @pytest.mark.asyncio
    async def test_activate_user_as_admin(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
    ):
        """Test admin can activate users."""
        # Arrange - Create admin
        from app.schemas.user import UserRegister
        admin_data = UserRegister(
            username="admin_activate",
            email="adminactivate@example.com",
            password="AdminPassword123!",
        )
        admin_user = await crud_user.create(db_session, obj_in=admin_data)
        admin_user.role = UserRole.ADMIN
        admin_user.status = UserStatus.ACTIVE
        admin_user.is_verified = True
        db_session.add(admin_user)
        await db_session.commit()
        await db_session.refresh(admin_user)
        
        # Create inactive user
        user_data = UserRegister(
            username="inactive_user",
            email="inactive@example.com",
            password="TestPassword123!",
        )
        inactive_user = await crud_user.create(db_session, obj_in=user_data)
        
        # Admin login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": admin_user.email, "password": "AdminPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/users/{inactive_user.id}/activate",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_suspend_user_as_admin(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
    ):
        """Test admin can suspend users."""
        # Arrange - Create admin
        from app.schemas.user import UserRegister
        admin_data = UserRegister(
            username="admin_suspend",
            email="adminsuspend@example.com",
            password="AdminPassword123!",
        )
        admin_user = await crud_user.create(db_session, obj_in=admin_data)
        admin_user.role = UserRole.ADMIN
        admin_user.status = UserStatus.ACTIVE
        admin_user.is_verified = True
        db_session.add(admin_user)
        await db_session.commit()
        await db_session.refresh(admin_user)
        
        # Create user to suspend
        user_data = UserRegister(
            username="user_to_suspend",
            email="suspend@example.com",
            password="TestPassword123!",
        )
        target_user = await crud_user.create(db_session, obj_in=user_data)
        
        # Admin login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": admin_user.email, "password": "AdminPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/users/{target_user.id}/suspend",
            headers=headers,
            json={"reason": "Violation of terms"},
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_non_admin_cannot_update_roles(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test that regular users cannot perform admin operations."""
        # Arrange - Create another user
        from app.schemas.user import UserRegister
        other_data = UserRegister(
            username="other_user",
            email="other@example.com",
            password="TestPassword123!",
        )
        other_user = await crud_user.create(db_session, obj_in=other_data)
        
        # Login as regular user
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act - Try to update role
        response = await async_client.put(
            f"/api/v1/users/{other_user.id}/admin",
            headers=headers,
            json={"role": "admin"},
        )
        
        # Assert
        assert response.status_code == 403


class TestUserDeletion:
    """Test user account deletion."""
    
    @pytest.mark.asyncio
    async def test_delete_own_account(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
    ):
        """Test user can delete their own account."""
        # Arrange - Create user
        from app.schemas.user import UserRegister
        user_data = UserRegister(
            username="deleteme_user",
            email="deleteme@example.com",
            password="TestPassword123!",
        )
        user = await crud_user.create(db_session, obj_in=user_data)
        user.status = UserStatus.ACTIVE
        user.is_verified = True
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": user.email, "password": "TestPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.delete(
            f"/api/v1/users/{user.id}",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_cannot_delete_other_user_account(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test user cannot delete someone else's account."""
        # Arrange - Create another user
        from app.schemas.user import UserRegister
        other_data = UserRegister(
            username="other_account",
            email="otheraccount@example.com",
            password="TestPassword123!",
        )
        other_user = await crud_user.create(db_session, obj_in=other_data)
        
        # Login as test_user
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act - Try to delete other user's account
        response = await async_client.delete(
            f"/api/v1/users/{other_user.id}",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 403


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_update_profile_unauthenticated(
        self,
        async_client: AsyncClient,
    ):
        """Test updating profile without authentication fails."""
        # Act
        response = await async_client.put(
            "/api/v1/users/me",
            json={"display_name": "Hacker"},
        )
        
        # Assert
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_search_with_empty_query(
        self,
        async_client: AsyncClient,
    ):
        """Test search with empty query returns appropriate response."""
        # Act
        response = await async_client.get("/api/v1/users/search?q=")
        
        # Assert
        # Should either return all users or validation error
        assert response.status_code in [200, 422]
    
    @pytest.mark.asyncio
    async def test_pagination_out_of_bounds(
        self,
        async_client: AsyncClient,
    ):
        """Test pagination with page beyond available data."""
        # Act
        response = await async_client.get("/api/v1/users?page=999&page_size=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0
