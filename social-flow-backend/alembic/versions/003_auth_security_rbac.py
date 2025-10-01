"""Add auth tokens, OAuth, 2FA, and RBAC tables

Revision ID: 003_auth_security_rbac
Revises: 002_add_performance_indexes
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_auth_security_rbac'
down_revision = '002_add_performance_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ========== PERMISSIONS TABLE ==========
    op.create_table(
        'permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('resource', sa.String(50), nullable=False, index=True),
        sa.Column('action', sa.String(50), nullable=False, index=True),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    # ========== ROLES TABLE ==========
    op.create_table(
        'roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('display_name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('priority', sa.String(10), default='0', nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('is_system', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    # ========== ROLE_PERMISSIONS TABLE (many-to-many) ==========
    op.create_table(
        'role_permissions',
        sa.Column('role_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('permission_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('permissions.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    # ========== USER_ROLES TABLE (many-to-many) ==========
    op.create_table(
        'user_roles',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    # ========== REFRESH_TOKENS TABLE ==========
    op.create_table(
        'refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('token', sa.String(500), unique=True, nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('token_family', sa.String(255), nullable=False, index=True),
        sa.Column('device_id', sa.String(255), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('is_revoked', sa.Boolean, default=False, nullable=False),
        sa.Column('is_used', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime, nullable=False),
        sa.Column('used_at', sa.DateTime, nullable=True),
        sa.Column('revoked_at', sa.DateTime, nullable=True),
    )
    
    # ========== TOKEN_BLACKLIST TABLE ==========
    op.create_table(
        'token_blacklist',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('jti', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('token', sa.Text, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('reason', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime, nullable=False),
    )
    
    # ========== OAUTH_ACCOUNTS TABLE ==========
    op.create_table(
        'oauth_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('provider', sa.String(50), nullable=False, index=True),
        sa.Column('provider_user_id', sa.String(255), nullable=False, index=True),
        sa.Column('provider_email', sa.String(255), nullable=True),
        sa.Column('provider_name', sa.String(255), nullable=True),
        sa.Column('provider_avatar', sa.String(500), nullable=True),
        sa.Column('access_token', sa.Text, nullable=True),
        sa.Column('refresh_token', sa.Text, nullable=True),
        sa.Column('token_expires_at', sa.DateTime, nullable=True),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_used_at', sa.DateTime, nullable=True),
    )
    
    # ========== TWO_FACTOR_AUTH TABLE ==========
    op.create_table(
        'two_factor_auth',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False, index=True),
        sa.Column('secret', sa.String(255), nullable=False),
        sa.Column('is_enabled', sa.Boolean, default=False, nullable=False),
        sa.Column('backup_codes', sa.Text, nullable=True),
        sa.Column('backup_codes_used', sa.Integer, default=0, nullable=False),
        sa.Column('is_verified', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('enabled_at', sa.DateTime, nullable=True),
        sa.Column('last_used_at', sa.DateTime, nullable=True),
    )
    
    # ========== INDEXES ==========
    # Additional composite indexes for performance
    op.create_index('idx_oauth_provider_user', 'oauth_accounts', ['provider', 'provider_user_id'], unique=True)
    op.create_index('idx_refresh_token_user_family', 'refresh_tokens', ['user_id', 'token_family'])
    op.create_index('idx_token_blacklist_expires', 'token_blacklist', ['expires_at'])
    op.create_index('idx_refresh_token_expires', 'refresh_tokens', ['expires_at'])
    
    # ========== SEED DEFAULT ROLES AND PERMISSIONS ==========
    # Insert default permissions
    op.execute("""
        INSERT INTO permissions (id, name, description, resource, action, is_active) VALUES
        (gen_random_uuid(), 'user:read', 'View user profiles', 'user', 'read', true),
        (gen_random_uuid(), 'user:update', 'Update own profile', 'user', 'update', true),
        (gen_random_uuid(), 'user:delete', 'Delete own account', 'user', 'delete', true),
        (gen_random_uuid(), 'user:ban', 'Ban users (admin only)', 'user', 'ban', true),
        (gen_random_uuid(), 'user:suspend', 'Suspend users (admin only)', 'user', 'suspend', true),
        (gen_random_uuid(), 'video:create', 'Upload videos', 'video', 'create', true),
        (gen_random_uuid(), 'video:read', 'Watch videos', 'video', 'read', true),
        (gen_random_uuid(), 'video:update', 'Edit own videos', 'video', 'update', true),
        (gen_random_uuid(), 'video:delete', 'Delete own videos', 'video', 'delete', true),
        (gen_random_uuid(), 'video:moderate', 'Moderate videos (moderator)', 'video', 'moderate', true),
        (gen_random_uuid(), 'post:create', 'Create posts', 'post', 'create', true),
        (gen_random_uuid(), 'post:read', 'Read posts', 'post', 'read', true),
        (gen_random_uuid(), 'post:update', 'Edit own posts', 'post', 'update', true),
        (gen_random_uuid(), 'post:delete', 'Delete own posts', 'post', 'delete', true),
        (gen_random_uuid(), 'post:moderate', 'Moderate posts (moderator)', 'post', 'moderate', true),
        (gen_random_uuid(), 'comment:create', 'Create comments', 'comment', 'create', true),
        (gen_random_uuid(), 'comment:read', 'Read comments', 'comment', 'read', true),
        (gen_random_uuid(), 'comment:update', 'Edit own comments', 'comment', 'update', true),
        (gen_random_uuid(), 'comment:delete', 'Delete own comments', 'comment', 'delete', true),
        (gen_random_uuid(), 'livestream:create', 'Start live streams', 'livestream', 'create', true),
        (gen_random_uuid(), 'livestream:read', 'Watch live streams', 'livestream', 'read', true),
        (gen_random_uuid(), 'livestream:moderate', 'Moderate live streams', 'livestream', 'moderate', true),
        (gen_random_uuid(), 'admin:all', 'Full admin access', 'admin', 'all', true)
    """)
    
    # Insert default roles
    op.execute("""
        INSERT INTO roles (id, name, display_name, description, priority, is_active, is_system) VALUES
        (gen_random_uuid(), 'admin', 'Administrator', 'Full system access with all permissions', '100', true, true),
        (gen_random_uuid(), 'moderator', 'Moderator', 'Content moderation permissions', '50', true, true),
        (gen_random_uuid(), 'creator', 'Content Creator', 'Can create and manage content', '20', true, true),
        (gen_random_uuid(), 'viewer', 'Viewer', 'Basic user permissions', '10', true, true)
    """)
    
    # Assign permissions to admin role (all permissions)
    op.execute("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r, permissions p
        WHERE r.name = 'admin'
    """)
    
    # Assign permissions to moderator role
    op.execute("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r, permissions p
        WHERE r.name = 'moderator'
        AND p.name IN (
            'user:read', 'user:suspend',
            'video:read', 'video:moderate',
            'post:read', 'post:moderate',
            'comment:read', 'comment:delete',
            'livestream:read', 'livestream:moderate'
        )
    """)
    
    # Assign permissions to creator role
    op.execute("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r, permissions p
        WHERE r.name = 'creator'
        AND p.name IN (
            'user:read', 'user:update', 'user:delete',
            'video:create', 'video:read', 'video:update', 'video:delete',
            'post:create', 'post:read', 'post:update', 'post:delete',
            'comment:create', 'comment:read', 'comment:update', 'comment:delete',
            'livestream:create', 'livestream:read'
        )
    """)
    
    # Assign permissions to viewer role (basic permissions)
    op.execute("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r, permissions p
        WHERE r.name = 'viewer'
        AND p.name IN (
            'user:read', 'user:update', 'user:delete',
            'video:read',
            'post:read',
            'comment:create', 'comment:read', 'comment:update', 'comment:delete',
            'livestream:read'
        )
    """)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_refresh_token_expires', table_name='refresh_tokens')
    op.drop_index('idx_token_blacklist_expires', table_name='token_blacklist')
    op.drop_index('idx_refresh_token_user_family', table_name='refresh_tokens')
    op.drop_index('idx_oauth_provider_user', table_name='oauth_accounts')
    
    # Drop tables (order matters due to foreign keys)
    op.drop_table('two_factor_auth')
    op.drop_table('oauth_accounts')
    op.drop_table('token_blacklist')
    op.drop_table('refresh_tokens')
    op.drop_table('user_roles')
    op.drop_table('role_permissions')
    op.drop_table('roles')
    op.drop_table('permissions')
