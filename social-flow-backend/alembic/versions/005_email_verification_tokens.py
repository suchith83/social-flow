"""Add EmailVerificationToken and PasswordResetToken tables.

Revision ID: 005_email_verification_tokens
Revises: 004_stripe_connect
Create Date: 2024-01-20 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005_email_verification_tokens'
down_revision: Union[str, None] = '004_stripe_connect'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create EmailVerificationToken and PasswordResetToken tables."""
    
    # ========================================================================
    # CREATE EMAIL_VERIFICATION_TOKENS TABLE
    # ========================================================================
    op.create_table(
        'email_verification_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('token', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('is_used', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign Key
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        
        # Indexes
        sa.Index('ix_email_verification_tokens_user_id', 'user_id'),
        sa.Index('ix_email_verification_tokens_expires_at', 'expires_at'),
        sa.Index('ix_email_verification_tokens_is_used', 'is_used'),
    )
    
    # ========================================================================
    # CREATE PASSWORD_RESET_TOKENS TABLE
    # ========================================================================
    op.create_table(
        'password_reset_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('token', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('is_used', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign Key
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        
        # Indexes
        sa.Index('ix_password_reset_tokens_user_id', 'user_id'),
        sa.Index('ix_password_reset_tokens_expires_at', 'expires_at'),
        sa.Index('ix_password_reset_tokens_is_used', 'is_used'),
    )
    
    # ========================================================================
    # ADD COMMENT TO TABLES
    # ========================================================================
    op.execute("""
        COMMENT ON TABLE email_verification_tokens IS 
        'Stores email verification tokens for user registration';
    """)
    
    op.execute("""
        COMMENT ON TABLE password_reset_tokens IS 
        'Stores password reset tokens for user password recovery';
    """)


def downgrade() -> None:
    """Drop EmailVerificationToken and PasswordResetToken tables."""
    op.drop_table('password_reset_tokens')
    op.drop_table('email_verification_tokens')
