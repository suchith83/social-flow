"""Full schema restructure

Revision ID: 006_full_schema_restructure
Revises: 005_email_verification_tokens
Create Date: 2025-10-01 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006_full_schema_restructure'
down_revision: Union[str, None] = '005_email_verification_tokens'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=False),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('website', sa.String(length=500), nullable=True),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('is_banned', sa.Boolean(), nullable=False),
        sa.Column('is_suspended', sa.Boolean(), nullable=False),
        sa.Column('ban_reason', sa.Text(), nullable=True),
        sa.Column('banned_at', sa.DateTime(), nullable=True),
        sa.Column('suspension_reason', sa.Text(), nullable=True),
        sa.Column('suspended_at', sa.DateTime(), nullable=True),
        sa.Column('suspension_ends_at', sa.DateTime(), nullable=True),
        sa.Column('followers_count', sa.Integer(), nullable=False),
        sa.Column('following_count', sa.Integer(), nullable=False),
        sa.Column('posts_count', sa.Integer(), nullable=False),
        sa.Column('videos_count', sa.Integer(), nullable=False),
        sa.Column('total_views', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.Column('last_activity_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )

    # Create roles table
    op.create_table('roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Create user_roles table
    op.create_table('user_roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.Column('assigned_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ),
        sa.ForeignKeyConstraint(['assigned_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create auth_tokens table
    op.create_table('auth_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token', sa.String(length=500), nullable=False),
        sa.Column('token_type', sa.String(length=20), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )

    # Create refresh_tokens table
    op.create_table('refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token', sa.String(length=500), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )

    # Create two_factor_auth table
    op.create_table('two_factor_auth',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('secret', sa.String(length=32), nullable=False),
        sa.Column('backup_codes', sa.Text(), nullable=True),
        sa.Column('is_enabled', sa.Boolean(), nullable=False),
        sa.Column('enabled_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )

    # Create subscriptions table
    op.create_table('subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stripe_subscription_id', sa.String(length=100), nullable=True),
        sa.Column('plan_name', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('current_period_start', sa.DateTime(), nullable=True),
        sa.Column('current_period_end', sa.DateTime(), nullable=True),
        sa.Column('cancel_at_period_end', sa.Boolean(), nullable=False),
        sa.Column('canceled_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create follows table
    op.create_table('follows',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('follower_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('following_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['follower_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['following_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('follower_id', 'following_id', name='unique_follow')
    )

    # Create videos table
    op.create_table('videos',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('video_url', sa.String(length=500), nullable=True),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('mime_type', sa.String(length=100), nullable=True),
        sa.Column('resolution', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('processing_progress', sa.Integer(), nullable=False),
        sa.Column('s3_key', sa.String(length=500), nullable=True),
        sa.Column('hls_playlist_url', sa.String(length=500), nullable=True),
        sa.Column('view_count', sa.Integer(), nullable=False),
        sa.Column('like_count', sa.Integer(), nullable=False),
        sa.Column('comment_count', sa.Integer(), nullable=False),
        sa.Column('share_count', sa.Integer(), nullable=False),
        sa.Column('is_private', sa.Boolean(), nullable=False),
        sa.Column('is_monetized', sa.Boolean(), nullable=False),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('copyright_status', sa.String(length=20), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('published_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create encoding_jobs table
    op.create_table('encoding_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('quality', sa.String(length=20), nullable=False),
        sa.Column('input_key', sa.String(length=500), nullable=False),
        sa.Column('output_key', sa.String(length=500), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create view_counts table
    op.create_table('view_counts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('watched_duration', sa.Integer(), nullable=False),
        sa.Column('is_completed', sa.Boolean(), nullable=False),
        sa.Column('viewed_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create posts table
    op.create_table('posts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('media_urls', sa.Text(), nullable=True),
        sa.Column('media_type', sa.String(length=20), nullable=True),
        sa.Column('is_repost', sa.Boolean(), nullable=False),
        sa.Column('original_post_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('repost_content', sa.Text(), nullable=True),
        sa.Column('is_approved', sa.Boolean(), nullable=False),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('approved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('like_count', sa.Integer(), nullable=False),
        sa.Column('repost_count', sa.Integer(), nullable=False),
        sa.Column('comment_count', sa.Integer(), nullable=False),
        sa.Column('share_count', sa.Integer(), nullable=False),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.Column('mentions', sa.Text(), nullable=True),
        sa.Column('hashtags', sa.Text(), nullable=True),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('is_pinned', sa.Boolean(), nullable=False),
        sa.Column('pinned_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['original_post_id'], ['posts.id'], ),
        sa.ForeignKeyConstraint(['approved_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create comments table
    op.create_table('comments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('post_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('parent_comment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('media_url', sa.String(length=500), nullable=True),
        sa.Column('media_type', sa.String(length=20), nullable=True),
        sa.Column('like_count', sa.Integer(), nullable=False),
        sa.Column('reply_count', sa.Integer(), nullable=False),
        sa.Column('is_approved', sa.Boolean(), nullable=False),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('approved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_pinned', sa.Boolean(), nullable=False),
        sa.Column('pinned_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['post_id'], ['posts.id'], ),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.ForeignKeyConstraint(['parent_comment_id'], ['comments.id'], ),
        sa.ForeignKeyConstraint(['author_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['approved_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create likes table
    op.create_table('likes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('post_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('comment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['post_id'], ['posts.id'], ),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ),
        sa.ForeignKeyConstraint(['comment_id'], ['comments.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'post_id', name='unique_post_like'),
        sa.UniqueConstraint('user_id', 'video_id', name='unique_video_like'),
        sa.UniqueConstraint('user_id', 'comment_id', name='unique_comment_like')
    )

    # Create live_streams table
    op.create_table('live_streams',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('stream_key', sa.String(length=100), nullable=False),
        sa.Column('rtmp_url', sa.String(length=500), nullable=False),
        sa.Column('hls_url', sa.String(length=500), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('is_private', sa.Boolean(), nullable=False),
        sa.Column('viewer_count', sa.Integer(), nullable=False),
        sa.Column('max_viewers', sa.Integer(), nullable=True),
        sa.Column('chat_enabled', sa.Boolean(), nullable=False),
        sa.Column('recording_enabled', sa.Boolean(), nullable=False),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('total_views', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stream_key')
    )

    # Create ads table
    op.create_table('ads',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('advertiser_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('video_url', sa.String(length=500), nullable=True),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=False),
        sa.Column('target_audience', sa.Text(), nullable=True),
        sa.Column('budget', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('spent', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('impressions', sa.Integer(), nullable=False),
        sa.Column('clicks', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['advertiser_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create payments table
    op.create_table('payments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('amount', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False),
        sa.Column('stripe_payment_intent_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('payment_type', sa.String(length=20), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create stripe_connect_accounts table
    op.create_table('stripe_connect_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stripe_account_id', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('capabilities', sa.Text(), nullable=True),
        sa.Column('requirements', sa.Text(), nullable=True),
        sa.Column('payouts_enabled', sa.Boolean(), nullable=False),
        sa.Column('charges_enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id'),
        sa.UniqueConstraint('stripe_account_id')
    )

    # Create notifications table
    op.create_table('notifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('data', sa.Text(), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=False),
        sa.Column('read_at', sa.DateTime(), nullable=True),
        sa.Column('priority', sa.String(length=20), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create analytics_events table
    op.create_table('analytics_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('event_data', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(length=100), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_users_username', 'users', ['username'], unique=False)
    op.create_index('ix_users_email', 'users', ['email'], unique=False)
    op.create_index('ix_users_created_at', 'users', ['created_at'], unique=False)
    op.create_index('ix_videos_owner_id', 'videos', ['owner_id'], unique=False)
    op.create_index('ix_videos_status', 'videos', ['status'], unique=False)
    op.create_index('ix_videos_created_at', 'videos', ['created_at'], unique=False)
    op.create_index('ix_posts_owner_id', 'posts', ['owner_id'], unique=False)
    op.create_index('ix_posts_created_at', 'posts', ['created_at'], unique=False)
    op.create_index('ix_comments_post_id', 'comments', ['post_id'], unique=False)
    op.create_index('ix_comments_video_id', 'comments', ['video_id'], unique=False)
    op.create_index('ix_follows_follower_id', 'follows', ['follower_id'], unique=False)
    op.create_index('ix_follows_following_id', 'follows', ['following_id'], unique=False)
    op.create_index('ix_live_streams_owner_id', 'live_streams', ['owner_id'], unique=False)
    op.create_index('ix_live_streams_status', 'live_streams', ['status'], unique=False)
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'], unique=False)
    op.create_index('ix_notifications_is_read', 'notifications', ['is_read'], unique=False)
    op.create_index('ix_analytics_events_user_id', 'analytics_events', ['user_id'], unique=False)
    op.create_index('ix_analytics_events_event_type', 'analytics_events', ['event_type'], unique=False)
    op.create_index('ix_analytics_events_timestamp', 'analytics_events', ['timestamp'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_analytics_events_timestamp', table_name='analytics_events')
    op.drop_index('ix_analytics_events_event_type', table_name='analytics_events')
    op.drop_index('ix_analytics_events_user_id', table_name='analytics_events')
    op.drop_index('ix_notifications_is_read', table_name='notifications')
    op.drop_index('ix_notifications_user_id', table_name='notifications')
    op.drop_index('ix_live_streams_status', table_name='live_streams')
    op.drop_index('ix_live_streams_owner_id', table_name='live_streams')
    op.drop_index('ix_follows_following_id', table_name='follows')
    op.drop_index('ix_follows_follower_id', table_name='follows')
    op.drop_index('ix_comments_video_id', table_name='comments')
    op.drop_index('ix_comments_post_id', table_name='comments')
    op.drop_index('ix_posts_created_at', table_name='posts')
    op.drop_index('ix_posts_owner_id', table_name='posts')
    op.drop_index('ix_videos_created_at', table_name='videos')
    op.drop_index('ix_videos_status', table_name='videos')
    op.drop_index('ix_videos_owner_id', table_name='videos')
    op.drop_index('ix_users_created_at', table_name='users')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_index('ix_users_username', table_name='users')

    # Drop tables
    op.drop_table('analytics_events')
    op.drop_table('notifications')
    op.drop_table('stripe_connect_accounts')
    op.drop_table('payments')
    op.drop_table('ads')
    op.drop_table('live_streams')
    op.drop_table('likes')
    op.drop_table('comments')
    op.drop_table('posts')
    op.drop_table('view_counts')
    op.drop_table('encoding_jobs')
    op.drop_table('videos')
    op.drop_table('follows')
    op.drop_table('subscriptions')
    op.drop_table('two_factor_auth')
    op.drop_table('refresh_tokens')
    op.drop_table('auth_tokens')
    op.drop_table('user_roles')
    op.drop_table('roles')
    op.drop_table('users')
