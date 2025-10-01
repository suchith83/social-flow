"""Add Stripe Connect tables for creator payouts.

Revision ID: 004_stripe_connect
Revises: 003_auth_security_rbac
Create Date: 2024-01-15 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004_stripe_connect'
down_revision: Union[str, None] = '003_auth_security_rbac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create Stripe Connect tables."""
    
    # ========================================================================
    # CREATE STRIPE_CONNECT_ACCOUNTS TABLE
    # ========================================================================
    op.create_table(
        'stripe_connect_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stripe_account_id', sa.String(255), nullable=False, unique=True),
        
        # Account Configuration
        sa.Column('account_type', sa.String(20), nullable=False, comment="express, standard, or custom"),
        sa.Column('country', sa.String(2), nullable=False, default='US'),
        sa.Column('currency', sa.String(3), nullable=False, default='usd'),
        
        # Account Status
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('charges_enabled', sa.Boolean, default=False),
        sa.Column('payouts_enabled', sa.Boolean, default=False),
        sa.Column('details_submitted', sa.Boolean, default=False),
        
        # Requirements (JSON arrays)
        sa.Column('requirements_currently_due', postgresql.JSONB, default=[]),
        sa.Column('requirements_eventually_due', postgresql.JSONB, default=[]),
        sa.Column('requirements_past_due', postgresql.JSONB, default=[]),
        
        # Business Information
        sa.Column('business_type', sa.String(50), nullable=True),
        sa.Column('business_name', sa.String(255), nullable=True),
        sa.Column('support_email', sa.String(255), nullable=True),
        sa.Column('support_phone', sa.String(50), nullable=True),
        
        # Banking Information
        sa.Column('external_account_id', sa.String(255), nullable=True),
        sa.Column('bank_name', sa.String(255), nullable=True),
        sa.Column('bank_last_four', sa.String(4), nullable=True),
        
        # Balance Tracking
        sa.Column('available_balance', sa.Numeric(12, 2), default=0.00),
        sa.Column('pending_balance', sa.Numeric(12, 2), default=0.00),
        sa.Column('total_volume', sa.Numeric(12, 2), default=0.00),
        
        # Onboarding
        sa.Column('onboarding_url', sa.Text, nullable=True),
        sa.Column('onboarding_completed_at', sa.DateTime(timezone=True), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Indexes for stripe_connect_accounts
    op.create_index('idx_stripe_connect_accounts_user_id', 'stripe_connect_accounts', ['user_id'], unique=True)
    op.create_index('idx_stripe_connect_accounts_stripe_account_id', 'stripe_connect_accounts', ['stripe_account_id'], unique=True)
    op.create_index('idx_stripe_connect_accounts_status', 'stripe_connect_accounts', ['status'])
    
    
    # ========================================================================
    # CREATE CREATOR_PAYOUTS TABLE
    # ========================================================================
    op.create_table(
        'creator_payouts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('connect_account_id', postgresql.UUID(as_uuid=True), nullable=False),
        
        # Payout Details
        sa.Column('amount', sa.Numeric(12, 2), nullable=False, comment="Total payout amount"),
        sa.Column('currency', sa.String(3), nullable=False, default='usd'),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        
        # Stripe References
        sa.Column('stripe_payout_id', sa.String(255), nullable=True, unique=True),
        sa.Column('stripe_transfer_id', sa.String(255), nullable=True),
        
        # Fee Breakdown
        sa.Column('gross_amount', sa.Numeric(12, 2), nullable=False, comment="Revenue before fees"),
        sa.Column('platform_fee', sa.Numeric(12, 2), nullable=False, default=0.00),
        sa.Column('stripe_fee', sa.Numeric(12, 2), nullable=False, default=0.00),
        sa.Column('net_amount', sa.Numeric(12, 2), nullable=False, comment="Amount after fees"),
        
        # Revenue Breakdown by Source
        sa.Column('subscription_revenue', sa.Numeric(12, 2), default=0.00),
        sa.Column('tips_revenue', sa.Numeric(12, 2), default=0.00),
        sa.Column('content_sales_revenue', sa.Numeric(12, 2), default=0.00),
        sa.Column('ad_revenue', sa.Numeric(12, 2), default=0.00),
        
        # Payout Period
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        
        # Additional Information
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('failure_message', sa.Text, nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_at', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['connect_account_id'], ['stripe_connect_accounts.id'], ondelete='CASCADE'),
    )
    
    # Indexes for creator_payouts
    op.create_index('idx_creator_payouts_user_id', 'creator_payouts', ['user_id'])
    op.create_index('idx_creator_payouts_connect_account_id', 'creator_payouts', ['connect_account_id'])
    op.create_index('idx_creator_payouts_stripe_payout_id', 'creator_payouts', ['stripe_payout_id'], unique=True)
    op.create_index('idx_creator_payouts_status', 'creator_payouts', ['status'])
    op.create_index('idx_creator_payouts_created_at', 'creator_payouts', ['created_at'])
    
    
    # ========================================================================
    # CREATE STRIPE_WEBHOOK_EVENTS TABLE
    # ========================================================================
    op.create_table(
        'stripe_webhook_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('stripe_event_id', sa.String(255), nullable=False, unique=True),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_data', postgresql.JSONB, nullable=False, comment="Full Stripe event payload"),
        sa.Column('event_version', sa.String(50), nullable=True),
        
        # Processing Status
        sa.Column('is_processed', sa.Boolean, default=False),
        sa.Column('processing_error', sa.Text, nullable=True),
        sa.Column('retry_count', sa.Integer, default=0),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    
    # Indexes for stripe_webhook_events
    op.create_index('idx_stripe_webhook_events_stripe_event_id', 'stripe_webhook_events', ['stripe_event_id'], unique=True)
    op.create_index('idx_stripe_webhook_events_event_type', 'stripe_webhook_events', ['event_type'])
    op.create_index('idx_stripe_webhook_events_is_processed', 'stripe_webhook_events', ['is_processed'])
    op.create_index('idx_stripe_webhook_events_created_at', 'stripe_webhook_events', ['created_at'])
    
    
    # ========================================================================
    # UPDATE PAYMENTS TABLE - Add connect_account_id
    # ========================================================================
    op.add_column(
        'payments',
        sa.Column('connect_account_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        'fk_payments_connect_account_id',
        'payments',
        'stripe_connect_accounts',
        ['connect_account_id'],
        ['id'],
        ondelete='SET NULL'
    )
    op.create_index('idx_payments_connect_account_id', 'payments', ['connect_account_id'])
    op.create_index('idx_payments_provider_payment_id', 'payments', ['provider_payment_id'])


def downgrade() -> None:
    """Drop Stripe Connect tables."""
    
    # Drop payments table additions
    op.drop_index('idx_payments_provider_payment_id', table_name='payments')
    op.drop_index('idx_payments_connect_account_id', table_name='payments')
    op.drop_constraint('fk_payments_connect_account_id', 'payments', type_='foreignkey')
    op.drop_column('payments', 'connect_account_id')
    
    # Drop stripe_webhook_events
    op.drop_index('idx_stripe_webhook_events_created_at', table_name='stripe_webhook_events')
    op.drop_index('idx_stripe_webhook_events_is_processed', table_name='stripe_webhook_events')
    op.drop_index('idx_stripe_webhook_events_event_type', table_name='stripe_webhook_events')
    op.drop_index('idx_stripe_webhook_events_stripe_event_id', table_name='stripe_webhook_events')
    op.drop_table('stripe_webhook_events')
    
    # Drop creator_payouts
    op.drop_index('idx_creator_payouts_created_at', table_name='creator_payouts')
    op.drop_index('idx_creator_payouts_status', table_name='creator_payouts')
    op.drop_index('idx_creator_payouts_stripe_payout_id', table_name='creator_payouts')
    op.drop_index('idx_creator_payouts_connect_account_id', table_name='creator_payouts')
    op.drop_index('idx_creator_payouts_user_id', table_name='creator_payouts')
    op.drop_table('creator_payouts')
    
    # Drop stripe_connect_accounts
    op.drop_index('idx_stripe_connect_accounts_status', table_name='stripe_connect_accounts')
    op.drop_index('idx_stripe_connect_accounts_stripe_account_id', table_name='stripe_connect_accounts')
    op.drop_index('idx_stripe_connect_accounts_user_id', table_name='stripe_connect_accounts')
    op.drop_table('stripe_connect_accounts')
