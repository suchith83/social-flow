"""
Payment and Monetization System Migration

Creates tables for comprehensive payment and monetization:
- subscriptions: User subscriptions with Stripe
- payouts: Creator earnings and payouts
- ad_campaigns: Advertisement campaigns
- ad_impressions: Ad view tracking
- donations: Livestream tips/donations
- revenue_splits: Automatic revenue sharing

Revision ID: payments_001
Revises: moderation_001
Create Date: 2024-01-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = 'payments_001'
down_revision = 'moderation_001'
branch_labels = None
depends_on = None


def upgrade():
    """Create payment and monetization tables"""
    
    # Create enum types
    subscription_tier_enum = postgresql.ENUM(
        'free', 'basic', 'premium', 'creator', 'enterprise',
        name='subscriptiontier',
        create_type=True
    )
    subscription_tier_enum.create(op.get_bind(), checkfirst=True)
    
    subscription_status_enum = postgresql.ENUM(
        'active', 'past_due', 'canceled', 'trialing', 
        'incomplete', 'incomplete_expired', 'unpaid',
        name='subscriptionstatus',
        create_type=True
    )
    subscription_status_enum.create(op.get_bind(), checkfirst=True)
    
    payout_status_enum = postgresql.ENUM(
        'pending', 'processing', 'completed', 'failed', 'canceled',
        name='payoutstatus',
        create_type=True
    )
    payout_status_enum.create(op.get_bind(), checkfirst=True)
    
    ad_campaign_status_enum = postgresql.ENUM(
        'draft', 'pending_approval', 'active', 'paused', 'completed', 'rejected',
        name='adcampaignstatus',
        create_type=True
    )
    ad_campaign_status_enum.create(op.get_bind(), checkfirst=True)
    
    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Subscription details
        sa.Column('tier', sa.String(20), nullable=False, default='free'),
        sa.Column('status', sa.String(20), nullable=False, default='active'),
        
        # Stripe integration
        sa.Column('stripe_customer_id', sa.String(100), nullable=True),
        sa.Column('stripe_subscription_id', sa.String(100), nullable=True),
        sa.Column('stripe_price_id', sa.String(100), nullable=True),
        
        # Pricing
        sa.Column('amount', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('billing_interval', sa.String(20), nullable=False, default='month'),
        
        # Trial
        sa.Column('trial_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trial_end', sa.DateTime(timezone=True), nullable=True),
        
        # Subscription period
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
        
        # Cancellation
        sa.Column('cancel_at_period_end', sa.Boolean, default=False, nullable=False),
        sa.Column('canceled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        
        # Features
        sa.Column('features', postgresql.JSONB, nullable=True, default={}),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True, default={}),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    
    # Create payouts table
    op.create_table(
        'payouts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('creator_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Payout details
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        
        # Amount
        sa.Column('amount', sa.Numeric(10, 2), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('fee', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('net_amount', sa.Numeric(10, 2), nullable=False),
        
        # Revenue breakdown
        sa.Column('ad_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('subscription_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('donation_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('other_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        
        # Payment provider
        sa.Column('provider', sa.String(50), nullable=False, default='stripe'),
        sa.Column('provider_payout_id', sa.String(100), nullable=True),
        sa.Column('payout_method', sa.String(50), nullable=True),
        
        # Period
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        
        # Bank details
        sa.Column('bank_account_last4', sa.String(4), nullable=True),
        
        # Failure info
        sa.Column('failure_code', sa.String(50), nullable=True),
        sa.Column('failure_message', sa.Text, nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True, default={}),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True)
    )
    
    # Create ad_campaigns table
    op.create_table(
        'ad_campaigns',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('advertiser_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Campaign details
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='draft'),
        
        # Targeting
        sa.Column('target_audience', postgresql.JSONB, nullable=True, default={}),
        sa.Column('target_placements', postgresql.JSONB, nullable=True, default=[]),
        
        # Budget
        sa.Column('daily_budget', sa.Numeric(10, 2), nullable=False),
        sa.Column('total_budget', sa.Numeric(10, 2), nullable=False),
        sa.Column('spent_amount', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        
        # Pricing model
        sa.Column('pricing_model', sa.String(20), nullable=False, default='cpm'),
        sa.Column('bid_amount', sa.Numeric(10, 4), nullable=False),
        
        # Creative assets
        sa.Column('creative_url', sa.String(500), nullable=False),
        sa.Column('creative_type', sa.String(20), nullable=False),
        sa.Column('click_url', sa.String(500), nullable=False),
        
        # Schedule
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        
        # Performance metrics
        sa.Column('impressions', sa.Integer, nullable=False, default=0),
        sa.Column('clicks', sa.Integer, nullable=False, default=0),
        sa.Column('views', sa.Integer, nullable=False, default=0),
        sa.Column('conversions', sa.Integer, nullable=False, default=0),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True, default={}),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # Constraints
        sa.CheckConstraint('daily_budget > 0', name='check_daily_budget_positive'),
        sa.CheckConstraint('total_budget > 0', name='check_total_budget_positive')
    )
    
    # Create ad_impressions table
    op.create_table(
        'ad_impressions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ad_campaigns.id', ondelete='CASCADE'), nullable=False),
        
        # Where ad was shown
        sa.Column('video_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=True),
        sa.Column('stream_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('live_streams.id', ondelete='CASCADE'), nullable=True),
        sa.Column('placement', sa.String(20), nullable=False),
        
        # Viewer info
        sa.Column('viewer_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('viewer_ip', sa.String(45), nullable=True),
        sa.Column('viewer_country', sa.String(2), nullable=True),
        sa.Column('viewer_device', sa.String(50), nullable=True),
        
        # Interaction
        sa.Column('was_clicked', sa.Boolean, default=False, nullable=False),
        sa.Column('was_viewed', sa.Boolean, default=False, nullable=False),
        sa.Column('view_duration', sa.Integer, nullable=True),
        
        # Cost
        sa.Column('cost', sa.Numeric(10, 4), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('clicked_at', sa.DateTime(timezone=True), nullable=True)
    )
    
    # Create donations table
    op.create_table(
        'donations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('donor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('recipient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('stream_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('live_streams.id', ondelete='CASCADE'), nullable=True),
        
        # Amount
        sa.Column('amount', sa.Numeric(10, 2), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, default='USD'),
        sa.Column('platform_fee', sa.Numeric(10, 2), nullable=False),
        sa.Column('net_amount', sa.Numeric(10, 2), nullable=False),
        
        # Message
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('is_anonymous', sa.Boolean, default=False, nullable=False),
        sa.Column('is_highlighted', sa.Boolean, default=False, nullable=False),
        
        # Payment provider
        sa.Column('provider', sa.String(50), nullable=False, default='stripe'),
        sa.Column('provider_transaction_id', sa.String(100), nullable=True),
        
        # Status
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('refunded', sa.Boolean, default=False, nullable=False),
        sa.Column('refunded_at', sa.DateTime(timezone=True), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True, default={}),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Constraints
        sa.CheckConstraint('amount > 0', name='check_donation_amount_positive')
    )
    
    # Create revenue_splits table
    op.create_table(
        'revenue_splits',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        
        # Content reference
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_type', sa.String(20), nullable=False),
        
        # Parties
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('collaborator_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Split percentage
        sa.Column('owner_percentage', sa.Numeric(5, 2), nullable=False),
        sa.Column('collaborator_percentage', sa.Numeric(5, 2), nullable=False),
        
        # Revenue tracking
        sa.Column('total_revenue', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('owner_earnings', sa.Numeric(10, 2), nullable=False, default=0),
        sa.Column('collaborator_earnings', sa.Numeric(10, 2), nullable=False, default=0),
        
        # Status
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # Constraints
        sa.CheckConstraint('owner_percentage + collaborator_percentage = 100', name='check_split_total_100'),
        sa.CheckConstraint('owner_percentage >= 0 AND owner_percentage <= 100', name='check_owner_percentage_range'),
        sa.CheckConstraint('collaborator_percentage >= 0 AND collaborator_percentage <= 100', name='check_collaborator_percentage_range')
    )
    
    # Create indexes for subscriptions
    op.create_index('idx_subscriptions_user', 'subscriptions', ['user_id'])
    op.create_index('idx_subscriptions_status', 'subscriptions', ['status'])
    op.create_index('idx_subscriptions_tier', 'subscriptions', ['tier'])
    op.create_index('idx_subscriptions_stripe_customer', 'subscriptions', ['stripe_customer_id'])
    op.create_index('idx_subscriptions_stripe_subscription', 'subscriptions', ['stripe_subscription_id'])
    op.create_index('idx_subscriptions_period_end', 'subscriptions', ['current_period_end'])
    
    # Create indexes for payouts
    op.create_index('idx_payouts_creator', 'payouts', ['creator_id'])
    op.create_index('idx_payouts_status', 'payouts', ['status'])
    op.create_index('idx_payouts_period', 'payouts', ['period_start', 'period_end'])
    op.create_index('idx_payouts_created', 'payouts', ['created_at'])
    op.create_index('idx_payouts_provider', 'payouts', ['provider_payout_id'])
    
    # Create indexes for ad_campaigns
    op.create_index('idx_ad_campaigns_advertiser', 'ad_campaigns', ['advertiser_id'])
    op.create_index('idx_ad_campaigns_status', 'ad_campaigns', ['status'])
    op.create_index('idx_ad_campaigns_dates', 'ad_campaigns', ['start_date', 'end_date'])
    
    # Create indexes for ad_impressions
    op.create_index('idx_ad_impressions_campaign', 'ad_impressions', ['campaign_id'])
    op.create_index('idx_ad_impressions_video', 'ad_impressions', ['video_id'])
    op.create_index('idx_ad_impressions_stream', 'ad_impressions', ['stream_id'])
    op.create_index('idx_ad_impressions_viewer', 'ad_impressions', ['viewer_id'])
    op.create_index('idx_ad_impressions_created', 'ad_impressions', ['created_at'])
    op.create_index('idx_ad_impressions_clicked', 'ad_impressions', ['was_clicked'])
    
    # Create indexes for donations
    op.create_index('idx_donations_donor', 'donations', ['donor_id'])
    op.create_index('idx_donations_recipient', 'donations', ['recipient_id'])
    op.create_index('idx_donations_stream', 'donations', ['stream_id'])
    op.create_index('idx_donations_created', 'donations', ['created_at'])
    op.create_index('idx_donations_status', 'donations', ['status'])
    
    # Create indexes for revenue_splits
    op.create_index('idx_revenue_splits_content', 'revenue_splits', ['content_id', 'content_type'])
    op.create_index('idx_revenue_splits_owner', 'revenue_splits', ['owner_id'])
    op.create_index('idx_revenue_splits_collaborator', 'revenue_splits', ['collaborator_id'])
    op.create_index('idx_revenue_splits_active', 'revenue_splits', ['is_active'])


def downgrade():
    """Drop payment and monetization tables"""
    
    # Drop indexes
    op.drop_index('idx_revenue_splits_active', 'revenue_splits')
    op.drop_index('idx_revenue_splits_collaborator', 'revenue_splits')
    op.drop_index('idx_revenue_splits_owner', 'revenue_splits')
    op.drop_index('idx_revenue_splits_content', 'revenue_splits')
    
    op.drop_index('idx_donations_status', 'donations')
    op.drop_index('idx_donations_created', 'donations')
    op.drop_index('idx_donations_stream', 'donations')
    op.drop_index('idx_donations_recipient', 'donations')
    op.drop_index('idx_donations_donor', 'donations')
    
    op.drop_index('idx_ad_impressions_clicked', 'ad_impressions')
    op.drop_index('idx_ad_impressions_created', 'ad_impressions')
    op.drop_index('idx_ad_impressions_viewer', 'ad_impressions')
    op.drop_index('idx_ad_impressions_stream', 'ad_impressions')
    op.drop_index('idx_ad_impressions_video', 'ad_impressions')
    op.drop_index('idx_ad_impressions_campaign', 'ad_impressions')
    
    op.drop_index('idx_ad_campaigns_dates', 'ad_campaigns')
    op.drop_index('idx_ad_campaigns_status', 'ad_campaigns')
    op.drop_index('idx_ad_campaigns_advertiser', 'ad_campaigns')
    
    op.drop_index('idx_payouts_provider', 'payouts')
    op.drop_index('idx_payouts_created', 'payouts')
    op.drop_index('idx_payouts_period', 'payouts')
    op.drop_index('idx_payouts_status', 'payouts')
    op.drop_index('idx_payouts_creator', 'payouts')
    
    op.drop_index('idx_subscriptions_period_end', 'subscriptions')
    op.drop_index('idx_subscriptions_stripe_subscription', 'subscriptions')
    op.drop_index('idx_subscriptions_stripe_customer', 'subscriptions')
    op.drop_index('idx_subscriptions_tier', 'subscriptions')
    op.drop_index('idx_subscriptions_status', 'subscriptions')
    op.drop_index('idx_subscriptions_user', 'subscriptions')
    
    # Drop tables
    op.drop_table('revenue_splits')
    op.drop_table('donations')
    op.drop_table('ad_impressions')
    op.drop_table('ad_campaigns')
    op.drop_table('payouts')
    op.drop_table('subscriptions')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS adcampaignstatus')
    op.execute('DROP TYPE IF EXISTS payoutstatus')
    op.execute('DROP TYPE IF EXISTS subscriptionstatus')
    op.execute('DROP TYPE IF EXISTS subscriptiontier')
