"""create livestream tables

Revision ID: livestream_001
Revises: copyright_system_001
Create Date: 2024-10-02 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON

# revision identifiers, used by Alembic.
revision = 'livestream_001'
down_revision = 'copyright_system_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create live streaming tables and indexes"""
    
    # Create enums
    op.execute("""
        CREATE TYPE streamstatus AS ENUM (
            'scheduled', 'starting', 'live', 'ending', 'ended', 'failed'
        )
    """)
    
    op.execute("""
        CREATE TYPE streamquality AS ENUM ('low', 'medium', 'high', 'ultra')
    """)
    
    op.execute("""
        CREATE TYPE chatmessagetype AS ENUM (
            'message', 'system', 'donation', 'subscription', 'moderator_action'
        )
    """)
    
    # Create live_streams table
    op.create_table(
        'live_streams',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Stream configuration
        sa.Column('stream_key', sa.String(64), unique=True, nullable=False, index=True),
        sa.Column('stream_url', sa.String(512)),
        sa.Column('playback_url', sa.String(512)),
        
        # Stream metadata
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('category', sa.String(50)),
        sa.Column('tags', JSON, default=list),
        sa.Column('thumbnail_url', sa.String(512)),
        
        # Stream status
        sa.Column('status', sa.String(20), nullable=False, default='scheduled', index=True),
        sa.Column('quality', sa.String(20), default='high'),
        
        # Streaming details
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('ended_at', sa.DateTime(timezone=True)),
        sa.Column('scheduled_start', sa.DateTime(timezone=True)),
        sa.Column('duration_seconds', sa.Integer, default=0),
        
        # Viewer metrics
        sa.Column('current_viewers', sa.Integer, default=0),
        sa.Column('peak_viewers', sa.Integer, default=0),
        sa.Column('total_views', sa.Integer, default=0),
        
        # Engagement metrics
        sa.Column('likes_count', sa.Integer, default=0),
        sa.Column('chat_messages_count', sa.Integer, default=0),
        
        # Monetization
        sa.Column('is_monetized', sa.Boolean, default=False),
        sa.Column('subscription_only', sa.Boolean, default=False),
        sa.Column('donation_enabled', sa.Boolean, default=True),
        sa.Column('total_revenue', sa.Float, default=0.0),
        
        # Recording
        sa.Column('is_recording', sa.Boolean, default=True),
        sa.Column('recording_url', sa.String(512)),
        sa.Column('recording_bucket', sa.String(100)),
        sa.Column('recording_key', sa.String(512)),
        
        # AWS MediaLive/IVS
        sa.Column('channel_id', sa.String(100)),
        sa.Column('ivs_channel_arn', sa.String(200)),
        sa.Column('ivs_playback_url', sa.String(512)),
        sa.Column('ivs_ingest_endpoint', sa.String(512)),
        
        # Moderation
        sa.Column('chat_enabled', sa.Boolean, default=True),
        sa.Column('chat_delay_seconds', sa.Integer, default=0),
        sa.Column('slow_mode_enabled', sa.Boolean, default=False),
        sa.Column('slow_mode_interval', sa.Integer, default=3),
        
        # Privacy
        sa.Column('is_public', sa.Boolean, default=True),
        sa.Column('is_unlisted', sa.Boolean, default=False),
        
        # Technical metrics
        sa.Column('bitrate_kbps', sa.Integer),
        sa.Column('framerate_fps', sa.Integer),
        sa.Column('resolution', sa.String(20)),
        sa.Column('codec', sa.String(20)),
        
        # Metadata
        sa.Column('metadata', JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True))
    )
    
    # Create stream_viewers table
    op.create_table(
        'stream_viewers',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('stream_id', UUID(as_uuid=True), sa.ForeignKey('live_streams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE')),
        
        # Viewer info
        sa.Column('session_id', sa.String(100), nullable=False, index=True),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.String(512)),
        
        # Viewing session
        sa.Column('joined_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), nullable=False),
        sa.Column('left_at', sa.DateTime(timezone=True)),
        sa.Column('watch_time_seconds', sa.Integer, default=0),
        
        # Quality metrics
        sa.Column('selected_quality', sa.String(20), default='high'),
        sa.Column('buffer_count', sa.Integer, default=0),
        
        # Metadata
        sa.Column('metadata', JSON, default=dict)
    )
    
    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('stream_id', UUID(as_uuid=True), sa.ForeignKey('live_streams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Message content
        sa.Column('message_type', sa.String(30), nullable=False, default='message'),
        sa.Column('content', sa.Text, nullable=False),
        
        # Message metadata
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('edited_at', sa.DateTime(timezone=True)),
        
        # Moderation
        sa.Column('is_deleted', sa.Boolean, default=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True)),
        sa.Column('deleted_by_user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL')),
        sa.Column('deletion_reason', sa.String(200)),
        
        # Toxicity detection
        sa.Column('is_flagged', sa.Boolean, default=False),
        sa.Column('toxicity_score', sa.Float),
        sa.Column('flagged_reason', sa.String(200)),
        
        # Donation/Subscription
        sa.Column('donation_amount', sa.Float),
        sa.Column('donation_currency', sa.String(3)),
        
        # Metadata
        sa.Column('metadata', JSON, default=dict)
    )
    
    # Create stream_recordings table
    op.create_table(
        'stream_recordings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('stream_id', UUID(as_uuid=True), sa.ForeignKey('live_streams.id', ondelete='CASCADE'), nullable=False),
        
        # Recording details
        sa.Column('recording_url', sa.String(512), nullable=False),
        sa.Column('bucket_name', sa.String(100), nullable=False),
        sa.Column('object_key', sa.String(512), nullable=False),
        
        # Recording metadata
        sa.Column('duration_seconds', sa.Integer, nullable=False),
        sa.Column('file_size_bytes', sa.Integer),
        sa.Column('format', sa.String(20), default='mp4'),
        sa.Column('resolution', sa.String(20)),
        sa.Column('bitrate_kbps', sa.Integer),
        
        # Processing status
        sa.Column('is_processed', sa.Boolean, default=False),
        sa.Column('processing_started_at', sa.DateTime(timezone=True)),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True)),
        
        # Availability
        sa.Column('is_available', sa.Boolean, default=True),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        
        # Metadata
        sa.Column('metadata', JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False)
    )
    
    # Create indexes for live_streams
    op.create_index('idx_stream_user_status', 'live_streams', ['user_id', 'status'])
    op.create_index('idx_stream_status_scheduled', 'live_streams', ['status', 'scheduled_start'])
    op.create_index('idx_stream_created', 'live_streams', ['created_at'])
    op.create_index('idx_stream_public', 'live_streams', ['is_public', 'status'])
    
    # Create indexes for stream_viewers
    op.create_index('idx_viewer_stream_active', 'stream_viewers', ['stream_id', 'left_at'])
    op.create_index('idx_viewer_user', 'stream_viewers', ['user_id'])
    op.create_index('idx_viewer_heartbeat', 'stream_viewers', ['last_heartbeat'])
    
    # Create indexes for chat_messages
    op.create_index('idx_chat_stream_time', 'chat_messages', ['stream_id', 'sent_at'])
    op.create_index('idx_chat_user', 'chat_messages', ['user_id'])
    op.create_index('idx_chat_flagged', 'chat_messages', ['is_flagged'])
    op.create_index('idx_chat_deleted', 'chat_messages', ['is_deleted'])
    
    # Create indexes for stream_recordings
    op.create_index('idx_recording_stream', 'stream_recordings', ['stream_id'])
    op.create_index('idx_recording_available', 'stream_recordings', ['is_available'])
    op.create_index('idx_recording_created', 'stream_recordings', ['created_at'])


def downgrade() -> None:
    """Drop live streaming tables and indexes"""
    
    # Drop tables
    op.drop_table('stream_recordings')
    op.drop_table('chat_messages')
    op.drop_table('stream_viewers')
    op.drop_table('live_streams')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS chatmessagetype')
    op.execute('DROP TYPE IF EXISTS streamquality')
    op.execute('DROP TYPE IF EXISTS streamstatus')
