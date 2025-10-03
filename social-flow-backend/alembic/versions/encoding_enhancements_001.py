"""Add encoding enhancements to encoding_jobs table

Revision ID: encoding_enhancements_001
Revises: previous_migration
Create Date: 2024-01-15 10:00:00.000000

This migration enhances the encoding_jobs table with:
- UUID primary keys
- HLS/DASH manifest URLs
- Output format and paths
- Enhanced status tracking
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'encoding_enhancements_001'
down_revision = None  # Update this to point to your last migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade encoding_jobs table."""
    
    # Check if table exists, if not create it
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    if 'encoding_jobs' not in inspector.get_table_names():
        # Create the table from scratch
        op.create_table(
            'encoding_jobs',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, index=True),
            sa.Column('video_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('input_path', sa.String(500), nullable=False),
            sa.Column('input_s3_key', sa.String(500), nullable=True),
            sa.Column('output_s3_prefix', sa.String(500), nullable=True),
            sa.Column('output_format', sa.String(50), default='hls'),
            sa.Column('output_paths', postgresql.JSON, nullable=True),
            sa.Column('hls_manifest_url', sa.String(500), nullable=True),
            sa.Column('dash_manifest_url', sa.String(500), nullable=True),
            sa.Column('mediaconvert_job_id', sa.String(200), nullable=True, index=True),
            sa.Column('status', sa.Enum('PENDING', 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELED', name='encodingstatus'), nullable=False, index=True),
            sa.Column('progress', sa.Integer, default=0),
            sa.Column('qualities', sa.Text, nullable=True),
            sa.Column('error_message', sa.Text, nullable=True),
            sa.Column('retry_count', sa.Integer, default=0),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        )
    else:
        # Alter existing table
        # Add new columns if they don't exist
        existing_columns = [col['name'] for col in inspector.get_columns('encoding_jobs')]
        
        if 'input_path' not in existing_columns:
            op.add_column('encoding_jobs', sa.Column('input_path', sa.String(500), nullable=True))
            # Copy data from input_s3_key to input_path
            op.execute("UPDATE encoding_jobs SET input_path = input_s3_key WHERE input_path IS NULL")
            # Make it non-nullable after data migration
            op.alter_column('encoding_jobs', 'input_path', nullable=False)
        
        if 'output_format' not in existing_columns:
            op.add_column('encoding_jobs', sa.Column('output_format', sa.String(50), server_default='hls'))
        
        if 'output_paths' not in existing_columns:
            op.add_column('encoding_jobs', sa.Column('output_paths', postgresql.JSON, nullable=True))
        
        if 'hls_manifest_url' not in existing_columns:
            op.add_column('encoding_jobs', sa.Column('hls_manifest_url', sa.String(500), nullable=True))
        
        if 'dash_manifest_url' not in existing_columns:
            op.add_column('encoding_jobs', sa.Column('dash_manifest_url', sa.String(500), nullable=True))
        
        if 'retry_count' not in existing_columns:
            op.add_column('encoding_jobs', sa.Column('retry_count', sa.Integer, server_default='0'))
        
        # Check if we need to update the status enum
        conn = op.get_bind()
        result = conn.execute(sa.text("SELECT unnest(enum_range(NULL::encodingstatus))::text"))
        existing_statuses = {row[0] for row in result}
        
        if 'PENDING' not in existing_statuses:
            # Need to add PENDING status
            op.execute("ALTER TYPE encodingstatus ADD VALUE IF NOT EXISTS 'PENDING' BEFORE 'QUEUED'")
        
        # Ensure ID column is UUID type (if migration from string ID needed)
        id_column = [col for col in inspector.get_columns('encoding_jobs') if col['name'] == 'id'][0]
        if not isinstance(id_column['type'], postgresql.UUID):
            # This is complex and requires careful data migration
            # For now, we'll log a warning
            print("WARNING: encoding_jobs.id is not UUID type. Manual migration may be required.")


def downgrade() -> None:
    """Downgrade encoding_jobs table."""
    
    # Remove added columns
    op.drop_column('encoding_jobs', 'dash_manifest_url')
    op.drop_column('encoding_jobs', 'hls_manifest_url')
    op.drop_column('encoding_jobs', 'output_paths')
    op.drop_column('encoding_jobs', 'output_format')
    op.drop_column('encoding_jobs', 'retry_count')
    
    # Note: Cannot easily remove enum values, so we leave the status enum as-is
