"""
Add search optimization indexes and extensions.

This migration adds:
- GIN indexes for full-text search on videos and posts
- Trigram indexes for fuzzy matching
- Composite indexes for common query patterns
- Materialized view for trending content

Revision ID: search_optimization_001
Revises: 
Create Date: 2025-10-02
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'search_optimization_001'
down_revision = None  # Update this to your latest migration
branch_labels = None
depends_on = None


def upgrade():
    """Apply search optimization indexes."""
    
    # Enable PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    op.execute('CREATE EXTENSION IF NOT EXISTS btree_gin')
    
    # Video search indexes
    # GIN index for trigram search on title
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_videos_title_trgm 
        ON videos USING gin (title gin_trgm_ops)
    ''')
    
    # GIN index for trigram search on description
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_videos_description_trgm 
        ON videos USING gin (description gin_trgm_ops)
    ''')
    
    # GIN index for tags (JSON/text search)
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_videos_tags_gin 
        ON videos USING gin (tags gin_trgm_ops)
    ''')
    
    # Composite index for filtered searches
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_videos_search_filtered 
        ON videos (visibility, status, is_approved, created_at DESC)
        WHERE visibility = 'public' AND status = 'processed' AND is_approved = true
    ''')
    
    # Composite index for trending video queries
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_videos_trending 
        ON videos (created_at DESC, views_count DESC, likes_count DESC)
        WHERE visibility = 'public' AND status = 'processed' AND is_approved = true
    ''')
    
    # Post search indexes
    # GIN index for trigram search on content
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_posts_content_trgm 
        ON posts USING gin (content gin_trgm_ops)
    ''')
    
    # GIN index for hashtags
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_posts_hashtags_gin 
        ON posts USING gin (hashtags gin_trgm_ops)
    ''')
    
    # Composite index for filtered post searches
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_posts_search_filtered 
        ON posts (is_approved, is_flagged, created_at DESC)
        WHERE is_approved = true AND is_flagged = false
    ''')
    
    # Composite index for trending posts
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_posts_trending 
        ON posts (created_at DESC, likes_count DESC, reposts_count DESC)
        WHERE is_approved = true AND is_flagged = false
    ''')
    
    # User search indexes
    # B-tree index for username prefix search
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_users_username_trgm 
        ON users USING gin (username gin_trgm_ops)
        WHERE is_active = true
    ''')
    
    # B-tree index for email prefix search
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_users_email_trgm 
        ON users USING gin (email gin_trgm_ops)
        WHERE is_active = true
    ''')
    
    # Materialized view for trending videos (refreshed periodically)
    op.execute('''
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_trending_videos AS
        SELECT 
            id,
            title,
            description,
            thumbnail_url,
            views_count,
            likes_count,
            comments_count,
            shares_count,
            owner_id,
            created_at,
            (views_count * 1.0 + likes_count * 5.0 + comments_count * 10.0 + shares_count * 15.0) * 
            (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0)) as trending_score
        FROM videos
        WHERE 
            visibility = 'public' 
            AND status = 'processed' 
            AND is_approved = true
            AND created_at >= NOW() - INTERVAL '30 days'
        ORDER BY trending_score DESC
        LIMIT 1000
    ''')
    
    # Index on materialized view
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_mv_trending_videos_score 
        ON mv_trending_videos (trending_score DESC)
    ''')
    
    # Materialized view for trending posts
    op.execute('''
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_trending_posts AS
        SELECT 
            id,
            content,
            media_url,
            hashtags,
            likes_count,
            reposts_count,
            comments_count,
            shares_count,
            owner_id,
            created_at,
            (likes_count * 1.0 + reposts_count * 3.0 + comments_count * 2.0 + shares_count * 2.5) * 
            (1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 43200.0)) as trending_score
        FROM posts
        WHERE 
            is_approved = true 
            AND is_flagged = false
            AND created_at >= NOW() - INTERVAL '7 days'
        ORDER BY trending_score DESC
        LIMIT 1000
    ''')
    
    # Index on materialized view
    op.execute('''
        CREATE INDEX IF NOT EXISTS idx_mv_trending_posts_score 
        ON mv_trending_posts (trending_score DESC)
    ''')
    
    print("Search optimization indexes created successfully")


def downgrade():
    """Remove search optimization indexes."""
    
    # Drop materialized views
    op.execute('DROP MATERIALIZED VIEW IF EXISTS mv_trending_videos')
    op.execute('DROP MATERIALIZED VIEW IF EXISTS mv_trending_posts')
    
    # Drop video indexes
    op.execute('DROP INDEX IF EXISTS idx_videos_title_trgm')
    op.execute('DROP INDEX IF EXISTS idx_videos_description_trgm')
    op.execute('DROP INDEX IF EXISTS idx_videos_tags_gin')
    op.execute('DROP INDEX IF EXISTS idx_videos_search_filtered')
    op.execute('DROP INDEX IF EXISTS idx_videos_trending')
    
    # Drop post indexes
    op.execute('DROP INDEX IF EXISTS idx_posts_content_trgm')
    op.execute('DROP INDEX IF EXISTS idx_posts_hashtags_gin')
    op.execute('DROP INDEX IF EXISTS idx_posts_search_filtered')
    op.execute('DROP INDEX IF EXISTS idx_posts_trending')
    
    # Drop user indexes
    op.execute('DROP INDEX IF EXISTS idx_users_username_trgm')
    op.execute('DROP INDEX IF EXISTS idx_users_email_trgm')
    
    print("Search optimization indexes removed")
