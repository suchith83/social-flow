"""
Content Moderation System Migration

Creates tables for AI-powered content moderation:
- moderation_results: Store AI analysis results
- content_flags: User-reported content
- moderation_rules: Configurable moderation thresholds
- moderation_actions: Audit trail of actions

Revision ID: moderation_001
Revises: livestream_001
Create Date: 2024-01-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = 'moderation_001'
down_revision = 'livestream_001'
branch_labels = None
depends_on = None


def upgrade():
    """Create moderation tables and indexes"""
    
    # Create enum types
    content_type_enum = postgresql.ENUM(
        'image', 'video', 'text', 'comment', 
        'profile_picture', 'live_stream', 'audio',
        name='contenttype',
        create_type=True
    )
    content_type_enum.create(op.get_bind(), checkfirst=True)
    
    moderation_status_enum = postgresql.ENUM(
        'pending', 'approved', 'rejected', 
        'auto_approved', 'pending_review',
        name='moderationstatus',
        create_type=True
    )
    moderation_status_enum.create(op.get_bind(), checkfirst=True)
    
    violation_type_enum = postgresql.ENUM(
        'nsfw', 'violence', 'hate_speech', 'harassment',
        'spam', 'underage', 'dangerous', 'misinformation',
        'copyright', 'other',
        name='violationtype',
        create_type=True
    )
    violation_type_enum.create(op.get_bind(), checkfirst=True)
    
    action_type_enum = postgresql.ENUM(
        'approve', 'reject', 'remove', 'blur',
        'age_restrict', 'demonetize', 'suspend_user',
        'warn_user', 'ban_user',
        name='actiontype',
        create_type=True
    )
    action_type_enum.create(op.get_bind(), checkfirst=True)
    
    # Create moderation_results table
    op.create_table(
        'moderation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('content_type', content_type_enum, nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_url', sa.String(500), nullable=True),
        sa.Column('content_text', sa.Text, nullable=True),
        
        # Violation flags
        sa.Column('is_nsfw', sa.Boolean, default=False, nullable=False),
        sa.Column('is_violent', sa.Boolean, default=False, nullable=False),
        sa.Column('is_hate_speech', sa.Boolean, default=False, nullable=False),
        sa.Column('is_harassment', sa.Boolean, default=False, nullable=False),
        sa.Column('is_spam', sa.Boolean, default=False, nullable=False),
        sa.Column('contains_underage', sa.Boolean, default=False, nullable=False),
        sa.Column('is_dangerous', sa.Boolean, default=False, nullable=False),
        
        # Scores (0.0 - 1.0)
        sa.Column('nsfw_score', sa.Float, default=0.0, nullable=False),
        sa.Column('violence_score', sa.Float, default=0.0, nullable=False),
        sa.Column('hate_speech_score', sa.Float, default=0.0, nullable=False),
        sa.Column('toxicity_score', sa.Float, default=0.0, nullable=False),
        sa.Column('spam_score', sa.Float, default=0.0, nullable=False),
        
        # Detected elements
        sa.Column('detected_labels', postgresql.JSONB, nullable=True),
        sa.Column('detected_text', postgresql.JSONB, nullable=True),
        sa.Column('detected_faces', postgresql.JSONB, nullable=True),
        sa.Column('detected_celebrities', postgresql.JSONB, nullable=True),
        
        # AI responses
        sa.Column('rekognition_response', postgresql.JSONB, nullable=True),
        sa.Column('comprehend_response', postgresql.JSONB, nullable=True),
        
        # Status
        sa.Column('status', moderation_status_enum, nullable=False),
        sa.Column('requires_review', sa.Boolean, default=False, nullable=False),
        sa.Column('reviewer_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('review_notes', sa.Text, nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reviewer_id'], ['users.id'], ondelete='SET NULL')
    )
    
    # Create content_flags table
    op.create_table(
        'content_flags',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('reporter_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_type', content_type_enum, nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('violation_type', violation_type_enum, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        
        # Status
        sa.Column('status', sa.String(20), default='pending', nullable=False),
        sa.Column('resolved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        
        # Link to moderation result
        sa.Column('moderation_result_id', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['reporter_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['resolved_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['moderation_result_id'], ['moderation_results.id'], ondelete='SET NULL')
    )
    
    # Create moderation_rules table
    op.create_table(
        'moderation_rules',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('violation_type', violation_type_enum, nullable=False),
        
        # Thresholds
        sa.Column('threshold_score', sa.Float, default=0.7, nullable=False),
        sa.Column('auto_action', action_type_enum, nullable=True),
        sa.Column('auto_action_score', sa.Float, default=0.9, nullable=False),
        
        # Configuration
        sa.Column('priority', sa.Integer, default=0, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    
    # Create moderation_actions table
    op.create_table(
        'moderation_actions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('moderation_result_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action_type', action_type_enum, nullable=False),
        sa.Column('taken_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_automated', sa.Boolean, default=False, nullable=False),
        sa.Column('reason', sa.Text, nullable=True),
        
        # Reversal
        sa.Column('is_reversed', sa.Boolean, default=False, nullable=False),
        sa.Column('reversed_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('reversed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reversal_reason', sa.Text, nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['moderation_result_id'], ['moderation_results.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['taken_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reversed_by'], ['users.id'], ondelete='SET NULL')
    )
    
    # Create indexes for moderation_results
    op.create_index('idx_moderation_results_content', 'moderation_results', ['content_type', 'content_id'])
    op.create_index('idx_moderation_results_user', 'moderation_results', ['user_id'])
    op.create_index('idx_moderation_results_status', 'moderation_results', ['status'])
    op.create_index('idx_moderation_results_review', 'moderation_results', ['requires_review'])
    op.create_index('idx_moderation_results_created', 'moderation_results', ['created_at'])
    op.create_index('idx_moderation_results_nsfw', 'moderation_results', ['is_nsfw'])
    op.create_index('idx_moderation_results_violent', 'moderation_results', ['is_violent'])
    
    # Create indexes for content_flags
    op.create_index('idx_content_flags_content', 'content_flags', ['content_type', 'content_id'])
    op.create_index('idx_content_flags_reporter', 'content_flags', ['reporter_id'])
    op.create_index('idx_content_flags_status', 'content_flags', ['status'])
    op.create_index('idx_content_flags_created', 'content_flags', ['created_at'])
    
    # Create indexes for moderation_rules
    op.create_index('idx_moderation_rules_violation', 'moderation_rules', ['violation_type'])
    op.create_index('idx_moderation_rules_active', 'moderation_rules', ['is_active'])
    op.create_index('idx_moderation_rules_priority', 'moderation_rules', ['priority'])
    
    # Create indexes for moderation_actions
    op.create_index('idx_moderation_actions_result', 'moderation_actions', ['moderation_result_id'])
    op.create_index('idx_moderation_actions_type', 'moderation_actions', ['action_type'])
    op.create_index('idx_moderation_actions_created', 'moderation_actions', ['created_at'])
    
    # Insert default moderation rules
    op.execute("""
        INSERT INTO moderation_rules (id, name, description, violation_type, threshold_score, auto_action, auto_action_score, priority, is_active)
        VALUES
            (gen_random_uuid(), 'NSFW Content', 'Detect and flag sexually explicit content', 'nsfw', 0.7, 'blur', 0.9, 100, true),
            (gen_random_uuid(), 'Violence', 'Detect violent or graphic content', 'violence', 0.7, 'age_restrict', 0.9, 90, true),
            (gen_random_uuid(), 'Hate Speech', 'Detect hate speech and discriminatory content', 'hate_speech', 0.7, 'remove', 0.9, 95, true),
            (gen_random_uuid(), 'Harassment', 'Detect harassment and bullying', 'harassment', 0.7, 'warn_user', 0.9, 85, true),
            (gen_random_uuid(), 'Spam', 'Detect spam and promotional content', 'spam', 0.8, 'remove', 0.95, 50, true),
            (gen_random_uuid(), 'Underage', 'Detect content involving minors', 'underage', 0.6, 'remove', 0.8, 100, true),
            (gen_random_uuid(), 'Dangerous Activities', 'Detect dangerous or harmful activities', 'dangerous', 0.7, 'age_restrict', 0.9, 80, true)
    """)


def downgrade():
    """Drop moderation tables and enums"""
    
    # Drop indexes
    op.drop_index('idx_moderation_actions_created', 'moderation_actions')
    op.drop_index('idx_moderation_actions_type', 'moderation_actions')
    op.drop_index('idx_moderation_actions_result', 'moderation_actions')
    
    op.drop_index('idx_moderation_rules_priority', 'moderation_rules')
    op.drop_index('idx_moderation_rules_active', 'moderation_rules')
    op.drop_index('idx_moderation_rules_violation', 'moderation_rules')
    
    op.drop_index('idx_content_flags_created', 'content_flags')
    op.drop_index('idx_content_flags_status', 'content_flags')
    op.drop_index('idx_content_flags_reporter', 'content_flags')
    op.drop_index('idx_content_flags_content', 'content_flags')
    
    op.drop_index('idx_moderation_results_violent', 'moderation_results')
    op.drop_index('idx_moderation_results_nsfw', 'moderation_results')
    op.drop_index('idx_moderation_results_created', 'moderation_results')
    op.drop_index('idx_moderation_results_review', 'moderation_results')
    op.drop_index('idx_moderation_results_status', 'moderation_results')
    op.drop_index('idx_moderation_results_user', 'moderation_results')
    op.drop_index('idx_moderation_results_content', 'moderation_results')
    
    # Drop tables
    op.drop_table('moderation_actions')
    op.drop_table('moderation_rules')
    op.drop_table('content_flags')
    op.drop_table('moderation_results')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS actiontype')
    op.execute('DROP TYPE IF EXISTS violationtype')
    op.execute('DROP TYPE IF EXISTS moderationstatus')
    op.execute('DROP TYPE IF EXISTS contenttype')
