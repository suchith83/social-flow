"""Analytics extended models migration

Revision ID: analytics_001
Revises: notifications_001
Create Date: 2024-10-02 14:30:00.000000

This migration adds comprehensive analytics tables for video metrics,
user behavior tracking, revenue reporting, and aggregated statistics.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'analytics_001'
down_revision = 'notifications_001'
branch_labels = None
depends_on = None


def upgrade():
    """Apply migration - create analytics tables."""
    
    # Create video_metrics table
    op.create_table(
        'video_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        
        # View metrics
        sa.Column('total_views', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('unique_views', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('views_24h', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('views_7d', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('views_30d', sa.BigInteger(), nullable=False, server_default='0'),
        
        # Watch time metrics
        sa.Column('total_watch_time', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('avg_watch_time', sa.Float(), nullable=False, server_default='0'),
        sa.Column('avg_watch_percentage', sa.Float(), nullable=False, server_default='0'),
        sa.Column('completion_rate', sa.Float(), nullable=False, server_default='0'),
        
        # Engagement metrics
        sa.Column('total_likes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_dislikes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_comments', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_shares', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_saves', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('like_rate', sa.Float(), nullable=False, server_default='0'),
        sa.Column('comment_rate', sa.Float(), nullable=False, server_default='0'),
        sa.Column('share_rate', sa.Float(), nullable=False, server_default='0'),
        
        # Retention and concurrent viewers
        sa.Column('retention_curve', postgresql.JSONB(), nullable=True),
        sa.Column('peak_concurrent_viewers', sa.Integer(), nullable=False, server_default='0'),
        
        # Traffic and device breakdowns
        sa.Column('traffic_sources', postgresql.JSONB(), nullable=True),
        sa.Column('referrer_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('device_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('os_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('browser_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('country_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('top_countries', postgresql.JSONB(), nullable=True),
        
        # Revenue metrics
        sa.Column('total_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('ad_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('subscription_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('donation_revenue', sa.Float(), nullable=False, server_default='0'),
        
        # Performance scores
        sa.Column('engagement_score', sa.Float(), nullable=False, server_default='0'),
        sa.Column('virality_score', sa.Float(), nullable=False, server_default='0'),
        sa.Column('quality_score', sa.Float(), nullable=False, server_default='0'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_calculated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        
        # Foreign key
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for video_metrics
    op.create_index('idx_video_metrics_video_id', 'video_metrics', ['video_id'])
    op.create_index('idx_video_metrics_updated_at', 'video_metrics', ['updated_at'])
    
    # Create user_behavior_metrics table
    op.create_table(
        'user_behavior_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        
        # Activity metrics
        sa.Column('total_sessions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_session_duration', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('avg_session_duration', sa.Float(), nullable=False, server_default='0'),
        sa.Column('sessions_24h', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('sessions_7d', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('sessions_30d', sa.Integer(), nullable=False, server_default='0'),
        
        # Content creation
        sa.Column('total_videos_uploaded', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_posts_created', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_comments_posted', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('videos_uploaded_30d', sa.Integer(), nullable=False, server_default='0'),
        
        # Content consumption
        sa.Column('total_videos_watched', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_watch_time', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('videos_watched_30d', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_daily_watch_time', sa.Float(), nullable=False, server_default='0'),
        
        # Engagement
        sa.Column('total_likes_given', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_comments_given', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_shares', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_follows', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_followers', sa.Integer(), nullable=False, server_default='0'),
        
        # Social metrics
        sa.Column('following_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('followers_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('followers_growth_30d', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('engagement_rate', sa.Float(), nullable=False, server_default='0'),
        
        # Creator metrics
        sa.Column('creator_status', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('total_video_views', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('total_video_likes', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('avg_video_performance', sa.Float(), nullable=False, server_default='0'),
        
        # Revenue metrics
        sa.Column('total_earnings', sa.Float(), nullable=False, server_default='0'),
        sa.Column('earnings_30d', sa.Float(), nullable=False, server_default='0'),
        sa.Column('subscription_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('ad_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('donation_revenue', sa.Float(), nullable=False, server_default='0'),
        
        # Spending metrics
        sa.Column('total_spent', sa.Float(), nullable=False, server_default='0'),
        sa.Column('spent_30d', sa.Float(), nullable=False, server_default='0'),
        sa.Column('subscriptions_purchased', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('donations_sent', sa.Float(), nullable=False, server_default='0'),
        
        # Preferences and patterns
        sa.Column('primary_device', sa.String(50), nullable=True),
        sa.Column('device_usage', postgresql.JSONB(), nullable=True),
        sa.Column('most_active_hours', postgresql.JSONB(), nullable=True),
        sa.Column('most_active_days', postgresql.JSONB(), nullable=True),
        sa.Column('favorite_categories', postgresql.JSONB(), nullable=True),
        sa.Column('watch_history_summary', postgresql.JSONB(), nullable=True),
        
        # User scores
        sa.Column('activity_score', sa.Float(), nullable=False, server_default='0'),
        sa.Column('creator_score', sa.Float(), nullable=False, server_default='0'),
        sa.Column('engagement_score', sa.Float(), nullable=False, server_default='0'),
        sa.Column('loyalty_score', sa.Float(), nullable=False, server_default='0'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_calculated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        
        # Foreign key
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for user_behavior_metrics
    op.create_index('idx_user_behavior_user_id', 'user_behavior_metrics', ['user_id'])
    op.create_index('idx_user_behavior_updated_at', 'user_behavior_metrics', ['updated_at'])
    op.create_index('idx_user_behavior_creator_status', 'user_behavior_metrics', ['creator_status'])
    
    # Create revenue_metrics table
    op.create_table(
        'revenue_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('period_type', sa.String(20), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Subscription revenue
        sa.Column('subscription_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('new_subscriptions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('renewed_subscriptions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('canceled_subscriptions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('active_subscriptions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('subscription_mrr', sa.Float(), nullable=False, server_default='0'),
        
        # Donation revenue
        sa.Column('donation_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('total_donations', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_donation_amount', sa.Float(), nullable=False, server_default='0'),
        sa.Column('unique_donors', sa.Integer(), nullable=False, server_default='0'),
        
        # Ad revenue
        sa.Column('ad_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('ad_impressions', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('ad_clicks', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('ad_ctr', sa.Float(), nullable=False, server_default='0'),
        sa.Column('ad_cpm', sa.Float(), nullable=False, server_default='0'),
        
        # Total revenue
        sa.Column('total_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('gross_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('net_revenue', sa.Float(), nullable=False, server_default='0'),
        sa.Column('platform_fee', sa.Float(), nullable=False, server_default='0'),
        
        # Payouts
        sa.Column('total_payouts', sa.Float(), nullable=False, server_default='0'),
        sa.Column('pending_payouts', sa.Float(), nullable=False, server_default='0'),
        sa.Column('completed_payouts', sa.Float(), nullable=False, server_default='0'),
        
        # Transactions
        sa.Column('total_transactions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('successful_transactions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_transactions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('refunded_transactions', sa.Integer(), nullable=False, server_default='0'),
        
        # User metrics
        sa.Column('paying_users', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('new_paying_users', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('churned_users', sa.Integer(), nullable=False, server_default='0'),
        
        # ARPU
        sa.Column('arpu', sa.Float(), nullable=False, server_default='0'),
        sa.Column('arppu', sa.Float(), nullable=False, server_default='0'),
        
        # Breakdowns
        sa.Column('revenue_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('revenue_by_country', postgresql.JSONB(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        
        # Foreign key
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for revenue_metrics
    op.create_index('idx_revenue_metrics_date', 'revenue_metrics', ['date'])
    op.create_index('idx_revenue_metrics_user_date', 'revenue_metrics', ['user_id', 'date'])
    op.create_index('idx_revenue_metrics_period_type', 'revenue_metrics', ['period_type'])
    
    # Create aggregated_metrics table
    op.create_table(
        'aggregated_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('aggregation_period', sa.String(20), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=False),
        sa.Column('end_date', sa.DateTime(), nullable=False),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('entity_type', sa.String(50), nullable=True),
        sa.Column('metrics_data', postgresql.JSONB(), nullable=False),
        sa.Column('total_count', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('total_value', sa.Float(), nullable=False, server_default='0'),
        sa.Column('avg_value', sa.Float(), nullable=False, server_default='0'),
        sa.Column('min_value', sa.Float(), nullable=True),
        sa.Column('max_value', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    
    # Create indexes for aggregated_metrics
    op.create_index('idx_aggregated_metrics_type_period', 'aggregated_metrics', ['metric_type', 'aggregation_period'])
    op.create_index('idx_aggregated_metrics_entity', 'aggregated_metrics', ['entity_id', 'entity_type'])
    op.create_index('idx_aggregated_metrics_start_date', 'aggregated_metrics', ['start_date'])
    
    # Create view_sessions table
    op.create_table(
        'view_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=False),
        
        # View details
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('video_duration', sa.Integer(), nullable=False),
        sa.Column('watch_percentage', sa.Float(), nullable=False, server_default='0'),
        sa.Column('completed', sa.Boolean(), nullable=False, server_default='false'),
        
        # Playback quality
        sa.Column('quality_level', sa.String(20), nullable=True),
        sa.Column('buffering_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('buffering_duration', sa.Integer(), nullable=False, server_default='0'),
        
        # Device and context
        sa.Column('device_type', sa.String(50), nullable=True),
        sa.Column('os', sa.String(50), nullable=True),
        sa.Column('browser', sa.String(50), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('country', sa.String(100), nullable=True),
        
        # Traffic source
        sa.Column('referrer', sa.String(500), nullable=True),
        sa.Column('traffic_source', sa.String(100), nullable=True),
        
        # Engagement
        sa.Column('liked', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('commented', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('shared', sa.Boolean(), nullable=False, server_default='false'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    )
    
    # Create indexes for view_sessions
    op.create_index('idx_view_sessions_video_id', 'view_sessions', ['video_id'])
    op.create_index('idx_view_sessions_user_id', 'view_sessions', ['user_id'])
    op.create_index('idx_view_sessions_created_at', 'view_sessions', ['created_at'])
    op.create_index('idx_view_sessions_session_id', 'view_sessions', ['session_id'])


def downgrade():
    """Revert migration - drop analytics tables."""
    op.drop_table('view_sessions')
    op.drop_table('aggregated_metrics')
    op.drop_table('revenue_metrics')
    op.drop_table('user_behavior_metrics')
    op.drop_table('video_metrics')
