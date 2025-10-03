"""RBAC (Role-Based Access Control) Tests."""

import pytest
from enum import Enum


class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"


class Permission(str, Enum):
    """System permissions."""
    CREATE_POST = "create_post"
    EDIT_POST = "edit_post"
    DELETE_POST = "delete_post"
    MODERATE_CONTENT = "moderate_content"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],
    Role.MODERATOR: [
        Permission.MODERATE_CONTENT,
        Permission.VIEW_ANALYTICS,
        Permission.EDIT_POST,
        Permission.DELETE_POST
    ],
    Role.USER: [
        Permission.CREATE_POST,
        Permission.EDIT_POST
    ],
    Role.GUEST: []
}


class TestAuthRBAC:
    """Test Role-Based Access Control."""

    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert len(admin_perms) == len(Permission)
        assert Permission.MANAGE_USERS in admin_perms
        assert Permission.MODERATE_CONTENT in admin_perms

    def test_user_limited_permissions(self):
        """Test that regular user has limited permissions."""
        user_perms = ROLE_PERMISSIONS[Role.USER]
        assert Permission.CREATE_POST in user_perms
        assert Permission.MANAGE_USERS not in user_perms
        assert Permission.MODERATE_CONTENT not in user_perms

    def test_guest_no_permissions(self):
        """Test that guest has no permissions."""
        guest_perms = ROLE_PERMISSIONS[Role.GUEST]
        assert len(guest_perms) == 0

    @pytest.mark.parametrize("role,permission,expected", [
        (Role.ADMIN, Permission.MANAGE_USERS, True),
        (Role.MODERATOR, Permission.MODERATE_CONTENT, True),
        (Role.USER, Permission.CREATE_POST, True),
        (Role.USER, Permission.MANAGE_USERS, False),
        (Role.GUEST, Permission.CREATE_POST, False),
    ])
    def test_has_permission(self, role, permission, expected):
        """Test permission checking for different roles."""
        has_permission = permission in ROLE_PERMISSIONS[role]
        assert has_permission == expected

    def test_role_hierarchy(self):
        """Test that role hierarchy is maintained."""
        admin_perms = set(ROLE_PERMISSIONS[Role.ADMIN])
        moderator_perms = set(ROLE_PERMISSIONS[Role.MODERATOR])
        user_perms = set(ROLE_PERMISSIONS[Role.USER])
        
        # Admin has all permissions (including moderator and user)
        assert moderator_perms.issubset(admin_perms)
        assert user_perms.issubset(admin_perms)
        
        # Moderator has at least EDIT_POST (shared with user)
        assert Permission.EDIT_POST in moderator_perms
        assert Permission.EDIT_POST in user_perms
        
        # User cannot moderate or manage
        assert Permission.MODERATE_CONTENT not in user_perms
        assert Permission.MANAGE_USERS not in user_perms

    def test_permission_check_function(self):
        """Test permission checking function."""
        def has_permission(user_role: Role, required_permission: Permission) -> bool:
            return required_permission in ROLE_PERMISSIONS[user_role]
        
        assert has_permission(Role.ADMIN, Permission.MANAGE_USERS) is True
        assert has_permission(Role.USER, Permission.MANAGE_USERS) is False

    def test_multiple_roles_scenario(self):
        """Test user with multiple roles (combined permissions)."""
        user_roles = [Role.USER, Role.MODERATOR]
        combined_perms = set()
        for role in user_roles:
            combined_perms.update(ROLE_PERMISSIONS[role])
        
        assert Permission.CREATE_POST in combined_perms
        assert Permission.MODERATE_CONTENT in combined_perms