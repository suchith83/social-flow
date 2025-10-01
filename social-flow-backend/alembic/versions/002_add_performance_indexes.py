"""Add performance indexes

Revision ID: 002
Revises: 001
Create Date: 2025-01-23

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance indexes for optimized queries."""
    
    # ==========================================
    # VIDEO MODEL INDEXES
    # ==========================================
    
    # Composite index for user's videos ordered by date
    # Used by: GET /api/v1/videos/user/{user_id}
    op.create_index(
        'idx_videos_owner_created',
        'videos',
        ['owner_id', 'created_at'],
        unique=False
    )
    
    # Composite index for filtering by status and visibility
    # Used by: Video discovery, moderation dashboards
    op.create_index(
        'idx_videos_status_visibility',
        'videos',
        ['status', 'visibility'],
        unique=False
    )
    
    # Index for trending/popular videos
    # Used by: GET /api/v1/videos/trending
    op.create_index(
        'idx_videos_views_count',
        'videos',
        ['views_count'],
        unique=False,
        postgresql_using='btree'
    )
    
    # Index for recent videos discovery
    # Used by: GET /api/v1/videos/recent
    op.create_index(
        'idx_videos_created_at',
        'videos',
        ['created_at'],
        unique=False
    )
    
    # Index for video duration filtering
    # Used by: Video search with duration filters
    op.create_index(
        'idx_videos_duration',
        'videos',
        ['duration'],
        unique=False
    )
    
    # ==========================================
    # POST MODEL INDEXES
    # ==========================================
    
    # Composite index for user's posts ordered by date
    # Used by: GET /api/v1/posts/user/{user_id}
    op.create_index(
        'idx_posts_owner_created',
        'posts',
        ['owner_id', 'created_at'],
        unique=False
    )
    
    # Index for chronological feed
    # Used by: GET /api/v1/posts/feed?algorithm=chronological
    op.create_index(
        'idx_posts_created_at',
        'posts',
        ['created_at'],
        unique=False
    )
    
    # Index for repost chains
    # Used by: Finding all reposts of a post
    op.create_index(
        'idx_posts_original_post_id',
        'posts',
        ['original_post_id'],
        unique=False,
        postgresql_where=sa.text('original_post_id IS NOT NULL')
    )
    
    # Index for engagement-based sorting
    # Used by: Feed ranking algorithms
    op.create_index(
        'idx_posts_likes_count',
        'posts',
        ['likes_count'],
        unique=False
    )
    
    # ==========================================
    # FOLLOW MODEL INDEXES
    # ==========================================
    
    # Unique composite index for follow relationships
    # Prevents duplicate follows, speeds up follow status checks
    op.create_index(
        'idx_follows_follower_following',
        'follows',
        ['follower_id', 'following_id'],
        unique=True
    )
    
    # Index for getting user's followers
    # Used by: GET /api/v1/users/{user_id}/followers
    op.create_index(
        'idx_follows_following_id',
        'follows',
        ['following_id'],
        unique=False
    )
    
    # Index for getting user's following list
    # Used by: GET /api/v1/users/{user_id}/following
    op.create_index(
        'idx_follows_follower_id',
        'follows',
        ['follower_id'],
        unique=False
    )
    
    # Index for recent follow activity
    op.create_index(
        'idx_follows_created_at',
        'follows',
        ['created_at'],
        unique=False
    )
    
    # ==========================================
    # COMMENT MODEL INDEXES
    # ==========================================
    
    # Index for video comments
    # Used by: GET /api/v1/videos/{video_id}/comments
    op.create_index(
        'idx_comments_video_id',
        'comments',
        ['video_id'],
        unique=False,
        postgresql_where=sa.text('video_id IS NOT NULL')
    )
    
    # Index for post comments
    # Used by: GET /api/v1/posts/{post_id}/comments
    op.create_index(
        'idx_comments_post_id',
        'comments',
        ['post_id'],
        unique=False,
        postgresql_where=sa.text('post_id IS NOT NULL')
    )
    
    # Composite index for user's comments ordered by date
    op.create_index(
        'idx_comments_owner_created',
        'comments',
        ['owner_id', 'created_at'],
        unique=False
    )
    
    # Index for nested comment threads
    op.create_index(
        'idx_comments_parent_id',
        'comments',
        ['parent_id'],
        unique=False,
        postgresql_where=sa.text('parent_id IS NOT NULL')
    )
    
    # ==========================================
    # LIKE MODEL INDEXES
    # ==========================================
    
    # Unique composite for video likes
    # Prevents duplicate likes
    op.create_index(
        'idx_likes_user_video',
        'likes',
        ['user_id', 'video_id'],
        unique=True,
        postgresql_where=sa.text('video_id IS NOT NULL')
    )
    
    # Unique composite for post likes
    op.create_index(
        'idx_likes_user_post',
        'likes',
        ['user_id', 'post_id'],
        unique=True,
        postgresql_where=sa.text('post_id IS NOT NULL')
    )
    
    # Index for video likes count
    op.create_index(
        'idx_likes_video_id',
        'likes',
        ['video_id'],
        unique=False,
        postgresql_where=sa.text('video_id IS NOT NULL')
    )
    
    # Index for post likes count
    op.create_index(
        'idx_likes_post_id',
        'likes',
        ['post_id'],
        unique=False,
        postgresql_where=sa.text('post_id IS NOT NULL')
    )
    
    # ==========================================
    # VIEW_COUNT MODEL INDEXES
    # ==========================================
    
    # Composite index for unique view tracking
    # Prevents duplicate view counts
    op.create_index(
        'idx_viewcounts_video_user',
        'view_counts',
        ['video_id', 'user_id'],
        unique=False
    )
    
    # Index for video view analytics
    op.create_index(
        'idx_viewcounts_video_created',
        'view_counts',
        ['video_id', 'created_at'],
        unique=False
    )
    
    # ==========================================
    # NOTIFICATION MODEL INDEXES
    # ==========================================
    
    # Composite index for user notifications
    # Used by: GET /api/v1/notifications
    op.create_index(
        'idx_notifications_user_created',
        'notifications',
        ['user_id', 'created_at'],
        unique=False
    )
    
    # Index for unread notifications
    op.create_index(
        'idx_notifications_user_read',
        'notifications',
        ['user_id', 'is_read'],
        unique=False
    )
    
    # ==========================================
    # AD MODEL INDEXES
    # ==========================================
    
    # Index for active ads
    op.create_index(
        'idx_ads_status',
        'ads',
        ['status'],
        unique=False
    )
    
    # Composite index for ad targeting queries
    op.create_index(
        'idx_ads_status_targeting',
        'ads',
        ['status', 'target_age_min', 'target_age_max'],
        unique=False,
        postgresql_where=sa.text("status = 'active'")
    )
    
    # Index for advertiser's ads
    op.create_index(
        'idx_ads_advertiser_id',
        'ads',
        ['advertiser_id'],
        unique=False
    )


def downgrade() -> None:
    """Remove performance indexes."""
    
    # Video indexes
    op.drop_index('idx_videos_owner_created', table_name='videos')
    op.drop_index('idx_videos_status_visibility', table_name='videos')
    op.drop_index('idx_videos_views_count', table_name='videos')
    op.drop_index('idx_videos_created_at', table_name='videos')
    op.drop_index('idx_videos_duration', table_name='videos')
    
    # Post indexes
    op.drop_index('idx_posts_owner_created', table_name='posts')
    op.drop_index('idx_posts_created_at', table_name='posts')
    op.drop_index('idx_posts_original_post_id', table_name='posts')
    op.drop_index('idx_posts_likes_count', table_name='posts')
    
    # Follow indexes
    op.drop_index('idx_follows_follower_following', table_name='follows')
    op.drop_index('idx_follows_following_id', table_name='follows')
    op.drop_index('idx_follows_follower_id', table_name='follows')
    op.drop_index('idx_follows_created_at', table_name='follows')
    
    # Comment indexes
    op.drop_index('idx_comments_video_id', table_name='comments')
    op.drop_index('idx_comments_post_id', table_name='comments')
    op.drop_index('idx_comments_owner_created', table_name='comments')
    op.drop_index('idx_comments_parent_id', table_name='comments')
    
    # Like indexes
    op.drop_index('idx_likes_user_video', table_name='likes')
    op.drop_index('idx_likes_user_post', table_name='likes')
    op.drop_index('idx_likes_video_id', table_name='likes')
    op.drop_index('idx_likes_post_id', table_name='likes')
    
    # ViewCount indexes
    op.drop_index('idx_viewcounts_video_user', table_name='view_counts')
    op.drop_index('idx_viewcounts_video_created', table_name='view_counts')
    
    # Notification indexes
    op.drop_index('idx_notifications_user_created', table_name='notifications')
    op.drop_index('idx_notifications_user_read', table_name='notifications')
    
    # Ad indexes
    op.drop_index('idx_ads_status', table_name='ads')
    op.drop_index('idx_ads_status_targeting', table_name='ads')
    op.drop_index('idx_ads_advertiser_id', table_name='ads')
