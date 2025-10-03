"""
Copyright detection and fingerprinting models.

This module defines models for copyright detection, content fingerprinting,
and automated revenue sharing.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, Numeric, Float, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class FingerprintType(str, Enum):
    """Enum for fingerprint types."""
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


class CopyrightFingerprint(Base):
    """Model for storing content fingerprints for copyright detection."""
    
    __tablename__ = "copyright_fingerprints"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Fingerprint data
    fingerprint_hash = Column(String(255), nullable=False, index=True, unique=True)
    fingerprint_type = Column(String(50), nullable=False)  # audio, video, image
    fingerprint_algorithm = Column(String(50), nullable=False)  # chromaprint, perceptual_hash, etc.
    fingerprint_data = Column(JSON, nullable=True)  # Detailed fingerprint data
    
    # Content details
    content_type = Column(String(50), nullable=False)  # video, audio, image
    content_title = Column(String(500), nullable=True)
    content_description = Column(Text, nullable=True)
    content_duration = Column(Float, nullable=True)  # Duration in seconds
    content_url = Column(String(1000), nullable=True)
    
    # Rights holder information
    rights_holder_name = Column(String(255), nullable=False)
    rights_holder_email = Column(String(255), nullable=True)
    rights_holder_organization = Column(String(255), nullable=True)
    
    # Copyright details
    copyright_year = Column(Integer, nullable=True)
    copyright_jurisdiction = Column(String(100), nullable=True)
    license_type = Column(String(100), nullable=True)
    
    # Detection settings
    match_threshold = Column(Float, default=0.85, nullable=False)  # 85% similarity threshold
    min_match_duration = Column(Float, default=7.0, nullable=False)  # 7 seconds minimum
    is_active = Column(Boolean, default=True, nullable=False)
    auto_claim = Column(Boolean, default=True, nullable=False)
    
    # Revenue sharing
    revenue_share_percentage = Column(Numeric(5, 2), default=Decimal('100.00'), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Foreign keys
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # Relationships
    owner = relationship("User", backref="copyright_fingerprints")
    matches = relationship("CopyrightMatch", back_populates="fingerprint", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<CopyrightFingerprint(id={self.id}, hash={self.fingerprint_hash[:16]}..., owner_id={self.owner_id})>"


class CopyrightMatch(Base):
    """Model for tracking copyright matches in uploaded content."""
    
    __tablename__ = "copyright_matches"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Match details
    match_score = Column(Float, nullable=False)  # Similarity score 0-1
    match_duration = Column(Float, nullable=False)  # Duration of matched content in seconds
    match_start_time = Column(Float, nullable=False)  # Start time in uploaded content
    match_end_time = Column(Float, nullable=False)  # End time in uploaded content
    
    # Reference content timing
    reference_start_time = Column(Float, nullable=True)  # Start time in reference content
    reference_end_time = Column(Float, nullable=True)  # End time in reference content
    
    # Match metadata
    match_metadata = Column(JSON, nullable=True)  # Additional match details
    
    # Status
    status = Column(
        String(50),
        default='pending_review',
        nullable=False
    )  # pending_review, confirmed, disputed, resolved, rejected
    
    # Actions taken
    action_taken = Column(
        String(50),
        nullable=True
    )  # block, monetize, track, manual_review
    
    is_monetized = Column(Boolean, default=False, nullable=False)
    is_blocked = Column(Boolean, default=False, nullable=False)
    
    # Revenue sharing (if monetized)
    revenue_split_percentage = Column(Numeric(5, 2), nullable=True)
    revenue_earned = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    revenue_paid = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    
    # Review information
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(UUID(as_uuid=True), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Dispute information
    is_disputed = Column(Boolean, default=False, nullable=False)
    disputed_at = Column(DateTime, nullable=True)
    dispute_reason = Column(Text, nullable=True)
    dispute_resolved_at = Column(DateTime, nullable=True)
    dispute_resolution = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Foreign keys
    fingerprint_id = Column(
        UUID(as_uuid=True),
        ForeignKey('copyright_fingerprints.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    uploader_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # Relationships
    fingerprint = relationship("CopyrightFingerprint", back_populates="matches")
    video = relationship("Video", backref="copyright_matches")
    uploader = relationship("User", backref="copyright_matches_as_uploader")
    
    def __repr__(self) -> str:
        return f"<CopyrightMatch(id={self.id}, score={self.match_score:.2f}, duration={self.match_duration:.1f}s, status={self.status})>"
    
    @property
    def is_significant_match(self) -> bool:
        """Check if match meets minimum duration threshold (7 seconds)."""
        return self.match_duration >= 7.0
    
    @property
    def match_percentage(self) -> float:
        """Calculate match percentage."""
        return self.match_score * 100


class Copyright(Base):
    """Model for copyright claims and management."""
    
    __tablename__ = "copyrights"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Claim details
    claim_type = Column(String(50), nullable=False)  # manual, automatic, system_detected
    claim_title = Column(String(500), nullable=False)
    claim_description = Column(Text, nullable=True)
    
    # Content information
    original_content_url = Column(String(1000), nullable=True)
    infringing_content_url = Column(String(1000), nullable=True)
    
    # Evidence
    evidence_urls = Column(JSON, nullable=True)  # Array of evidence URLs
    evidence_description = Column(Text, nullable=True)
    
    # Status
    status = Column(
        String(50),
        default='pending',
        nullable=False
    )  # pending, under_review, approved, rejected, withdrawn
    
    # Resolution
    resolution = Column(String(50), nullable=True)  # takedown, monetize, license, dismiss
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(UUID(as_uuid=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Foreign keys
    claimant_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    alleged_infringer_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True
    )
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='CASCADE'),
        nullable=True,
        index=True
    )
    
    # Relationships
    claimant = relationship("User", foreign_keys=[claimant_id], backref="copyright_claims_filed")
    alleged_infringer = relationship("User", foreign_keys=[alleged_infringer_id], backref="copyright_claims_against")
    video = relationship("Video", backref="copyright_claims")
    
    def __repr__(self) -> str:
        return f"<Copyright(id={self.id}, title={self.claim_title[:50]}, status={self.status})>"
