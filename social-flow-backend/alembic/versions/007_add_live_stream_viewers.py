"""Add live stream viewers table

Revision ID: 007_add_live_stream_viewers
Revises: 006_full_schema_restructure
Create Date: 2025-01-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '007_add_live_stream_viewers'
down_revision: Union[str, None] = '006_full_schema_restructure'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create live_stream_viewers table
    op.create_table('live_stream_viewers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('live_stream_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('joined_at', sa.DateTime(), nullable=False),
        sa.Column('left_at', sa.DateTime(), nullable=True),
        sa.Column('watch_duration', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['live_stream_id'], ['live_streams.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('live_stream_id', 'user_id', name='unique_live_stream_viewer')
    )

    # Create indexes
    op.create_index('ix_live_stream_viewers_live_stream_id', 'live_stream_viewers', ['live_stream_id'], unique=False)
    op.create_index('ix_live_stream_viewers_user_id', 'live_stream_viewers', ['user_id'], unique=False)
    op.create_index('ix_live_stream_viewers_is_active', 'live_stream_viewers', ['is_active'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_live_stream_viewers_is_active', table_name='live_stream_viewers')
    op.drop_index('ix_live_stream_viewers_user_id', table_name='live_stream_viewers')
    op.drop_index('ix_live_stream_viewers_live_stream_id', table_name='live_stream_viewers')

    # Drop table
    op.drop_table('live_stream_viewers')