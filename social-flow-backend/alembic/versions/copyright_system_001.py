"""Add copyright fingerprint and match tables

Revision ID: copyright_system_001
Revises: encoding_enhancements_001
Create Date: 2024-01-15 14:00:00.000000

This migration creates tables for copyright detection:
- copyright_fingerprints: Stores audio/video fingerprints
- copyright_matches: Tracks detected copyright matches
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'copyright_system_001'
down_revision = 'encoding_enhancements_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create copyright detection tables."""
    
    # Create content_type enum
    op.execute("""
        CREATE TYPE contenttype AS ENUM (
            'music', 'video', 'movie', 'tv_show', 'podcast', 'other'
        )
    """)
    
    # Create fingerprint_type enum
    op.execute("""
        CREATE TYPE fingerprinttype AS ENUM (
            'audio', 'video', 'combined'
        )
    """)
    
    # Create copyright_fingerprints table
    op.create_table(
        'copyright_fingerprints',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, index=True),
        sa.Column('content_title', sa.String(500), nullable=False, index=True),
        sa.Column('content_artist', sa.String(300), nullable=True, index=True),
        sa.Column('content_type', sa.Enum('music', 'video', 'movie', 'tv_show', 'podcast', 'other', name='contenttype'), nullable=False, index=True),
        sa.Column('fingerprint_type', sa.Enum('audio', 'video', 'combined', name='fingerprinttype'), nullable=False, index=True),
        
        # Fingerprint data
        sa.Column('audio_fingerprint', sa.Text, nullable=True),
        sa.Column('audio_duration_seconds', sa.Float, nullable=True),
        sa.Column('video_hash', sa.Text, nullable=True),
        sa.Column('video_duration_seconds', sa.Float, nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSON, nullable=True),
        
        # Rights holder
        sa.Column('rights_holder_name', sa.String(300), nullable=True),
        sa.Column('rights_holder_id', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Revenue configuration
        sa.Column('revenue_share_percentage', sa.Float, default=100.0),
        sa.Column('block_content', sa.Boolean, default=False),
        sa.Column('match_threshold', sa.Float, default=85.0),
        
        # Source
        sa.Column('source_url', sa.String(500), nullable=True),
        sa.Column('external_id', sa.String(200), nullable=True, index=True),
        
        # Status
        sa.Column('is_active', sa.Boolean, default=True, index=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create composite indexes
    op.create_index(
        'idx_content_title_artist',
        'copyright_fingerprints',
        ['content_title', 'content_artist']
    )
    
    op.create_index(
        'idx_fingerprint_type_active',
        'copyright_fingerprints',
        ['fingerprint_type', 'is_active']
    )
    
    op.create_index(
        'idx_content_type_active',
        'copyright_fingerprints',
        ['content_type', 'is_active']
    )
    
    # Create copyright_matches table
    op.create_table(
        'copyright_matches',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, index=True),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('fingerprint_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        
        # Match details
        sa.Column('match_score', sa.Float, nullable=False),
        sa.Column('match_type', sa.Enum('audio', 'video', 'combined', name='fingerprinttype'), nullable=False),
        sa.Column('match_segments', postgresql.JSON, nullable=True),
        sa.Column('matched_duration', sa.Float, nullable=True),
        sa.Column('total_duration', sa.Float, nullable=True),
        
        # Actions
        sa.Column('action_taken', sa.String(100), nullable=True),
        sa.Column('revenue_split_percentage', sa.Float, nullable=True),
        
        # Review
        sa.Column('reviewed', sa.Boolean, default=False),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reviewed_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_false_positive', sa.Boolean, default=False),
        sa.Column('notes', sa.Text, nullable=True),
        
        # Timestamp
        sa.Column('detected_at', sa.DateTime(timezone=True), nullable=False, index=True),
    )
    
    # Create indexes on copyright_matches
    op.create_index(
        'idx_video_fingerprint',
        'copyright_matches',
        ['video_id', 'fingerprint_id']
    )
    
    op.create_index(
        'idx_match_score',
        'copyright_matches',
        ['match_score']
    )
    
    # Add foreign key constraints
    op.create_foreign_key(
        'fk_copyright_matches_video_id',
        'copyright_matches', 'videos',
        ['video_id'], ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_copyright_matches_fingerprint_id',
        'copyright_matches', 'copyright_fingerprints',
        ['fingerprint_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    """Drop copyright detection tables."""
    
    # Drop foreign keys first
    op.drop_constraint('fk_copyright_matches_fingerprint_id', 'copyright_matches', type_='foreignkey')
    op.drop_constraint('fk_copyright_matches_video_id', 'copyright_matches', type_='foreignkey')
    
    # Drop indexes
    op.drop_index('idx_match_score', table_name='copyright_matches')
    op.drop_index('idx_video_fingerprint', table_name='copyright_matches')
    op.drop_index('idx_content_type_active', table_name='copyright_fingerprints')
    op.drop_index('idx_fingerprint_type_active', table_name='copyright_fingerprints')
    op.drop_index('idx_content_title_artist', table_name='copyright_fingerprints')
    
    # Drop tables
    op.drop_table('copyright_matches')
    op.drop_table('copyright_fingerprints')
    
    # Drop enums
    op.execute('DROP TYPE fingerprinttype')
    op.execute('DROP TYPE contenttype')
