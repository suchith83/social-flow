"""
Notification System Migration

Creates tables for enhanced notification system:
- notification_preferences: User notification settings
- notification_templates: Reusable templates
- email_logs: Email delivery tracking
- push_notification_tokens: Mobile device tokens

Extends existing notifications table

Revision ID: notifications_001
Revises: payments_001
Create Date: 2024-10-02
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = 'notifications_001'
down_revision = 'payments_001'
branch_labels = None
depends_on = None


def upgrade():
    """Create enhanced notification system tables"""
    
    # Create notification_preferences table
    op.create_table(
        'notification_preferences',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True),
        
        # Global settings
        sa.Column('email_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('push_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('sms_enabled', sa.Boolean, default=False, nullable=False),
        
        # Engagement notifications
        sa.Column('new_follower_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('new_like_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('new_comment_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('mention_enabled', sa.Boolean, default=True, nullable=False),
        
        # Content notifications
        sa.Column('video_processing_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('live_stream_enabled', sa.Boolean, default=True, nullable=False),
        
        # Moderation notifications
        sa.Column('moderation_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('copyright_enabled', sa.Boolean, default=True, nullable=False),
        
        # Payment notifications
        sa.Column('payment_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('payout_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('donation_enabled', sa.Boolean, default=True, nullable=False),
        
        # System notifications
        sa.Column('system_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('security_enabled', sa.Boolean, default=True, nullable=False),
        
        # Digest settings
        sa.Column('daily_digest_enabled', sa.Boolean, default=False, nullable=False),
        sa.Column('weekly_digest_enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('digest_time', sa.String(5), default='09:00', nullable=False),
        
        # Quiet hours
        sa.Column('quiet_hours_enabled', sa.Boolean, default=False, nullable=False),
        sa.Column('quiet_hours_start', sa.String(5), nullable=True),
        sa.Column('quiet_hours_end', sa.String(5), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    
    # Create notification_templates table
    op.create_table(
        'notification_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        
        # Template details
        sa.Column('type', sa.String(50), nullable=False, unique=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        
        # Content templates
        sa.Column('title_template', sa.String(200), nullable=False),
        sa.Column('message_template', sa.Text, nullable=False),
        
        # Email templates
        sa.Column('email_subject_template', sa.String(200), nullable=True),
        sa.Column('email_body_template', sa.Text, nullable=True),
        sa.Column('email_html_template', sa.Text, nullable=True),
        
        # Push notification templates
        sa.Column('push_title_template', sa.String(200), nullable=True),
        sa.Column('push_body_template', sa.Text, nullable=True),
        
        # Default values
        sa.Column('default_icon', sa.String(100), nullable=True),
        sa.Column('default_action_label', sa.String(100), nullable=True),
        sa.Column('default_priority', sa.String(20), default='normal', nullable=False),
        sa.Column('default_channels', postgresql.ARRAY(sa.String), default=['in_app'], nullable=False),
        
        # Expiry
        sa.Column('default_expires_hours', sa.Integer, nullable=True),
        
        # Status
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    
    # Create email_logs table
    op.create_table(
        'email_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        
        # Notification reference
        sa.Column('notification_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('notifications.id', ondelete='CASCADE'), nullable=True),
        
        # Recipient
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('to_email', sa.String(255), nullable=False),
        
        # Email details
        sa.Column('subject', sa.String(200), nullable=False),
        sa.Column('body_text', sa.Text, nullable=True),
        sa.Column('body_html', sa.Text, nullable=True),
        
        # Provider details
        sa.Column('provider', sa.String(50), default='sendgrid', nullable=False),
        sa.Column('provider_message_id', sa.String(200), nullable=True),
        
        # Status
        sa.Column('status', sa.String(20), default='pending', nullable=False),
        
        # Delivery tracking
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('opened_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('clicked_at', sa.DateTime(timezone=True), nullable=True),
        
        # Error tracking
        sa.Column('error_code', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('bounce_type', sa.String(50), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True, default={}),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )
    
    # Create push_notification_tokens table
    op.create_table(
        'push_notification_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        
        # User
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Device details
        sa.Column('device_id', sa.String(200), nullable=False),
        sa.Column('device_type', sa.String(20), nullable=False),
        sa.Column('device_name', sa.String(200), nullable=True),
        
        # Token
        sa.Column('token', sa.String(500), nullable=False, unique=True),
        
        # Status
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Usage tracking
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    
    # Create indexes for notification_preferences
    op.create_index('idx_notification_prefs_user', 'notification_preferences', ['user_id'])
    
    # Create indexes for notification_templates
    op.create_index('idx_notification_templates_type', 'notification_templates', ['type'])
    op.create_index('idx_notification_templates_active', 'notification_templates', ['is_active'])
    
    # Create indexes for email_logs
    op.create_index('idx_email_logs_user_created', 'email_logs', ['user_id', 'created_at'])
    op.create_index('idx_email_logs_status', 'email_logs', ['status'])
    op.create_index('idx_email_logs_provider_id', 'email_logs', ['provider_message_id'])
    
    # Create indexes for push_notification_tokens
    op.create_index('idx_push_tokens_user_active', 'push_notification_tokens', ['user_id', 'is_active'])
    op.create_index('idx_push_tokens_device', 'push_notification_tokens', ['device_id'])
    op.create_index('idx_push_tokens_token', 'push_notification_tokens', ['token'])


def downgrade():
    """Drop enhanced notification system tables"""
    
    # Drop indexes
    op.drop_index('idx_push_tokens_token', 'push_notification_tokens')
    op.drop_index('idx_push_tokens_device', 'push_notification_tokens')
    op.drop_index('idx_push_tokens_user_active', 'push_notification_tokens')
    
    op.drop_index('idx_email_logs_provider_id', 'email_logs')
    op.drop_index('idx_email_logs_status', 'email_logs')
    op.drop_index('idx_email_logs_user_created', 'email_logs')
    
    op.drop_index('idx_notification_templates_active', 'notification_templates')
    op.drop_index('idx_notification_templates_type', 'notification_templates')
    
    op.drop_index('idx_notification_prefs_user', 'notification_preferences')
    
    # Drop tables
    op.drop_table('push_notification_tokens')
    op.drop_table('email_logs')
    op.drop_table('notification_templates')
    op.drop_table('notification_preferences')
