#!/usr/bin/env python3
"""
Script to add UUID conversions to analytics service methods.
"""

import re

# Read the file
with open('app/analytics/services/enhanced_service.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the replacements - each is a tuple of (old_text, new_text)
replacements = [
    # calculate_video_metrics
    (
        '''    async def calculate_video_metrics(self, video_id: str) -> VideoMetrics:
        """Calculate comprehensive metrics for a video."""
        try:
            # Check if metrics exist
            stmt = select(VideoMetrics).where(VideoMetrics.video_id == video_id)''',
        '''    async def calculate_video_metrics(self, video_id: str) -> VideoMetrics:
        """Calculate comprehensive metrics for a video."""
        try:
            import uuid as uuid_module
            video_uuid = uuid_module.UUID(video_id) if isinstance(video_id, str) else video_id
            
            # Check if metrics exist
            stmt = select(VideoMetrics).where(VideoMetrics.video_id == video_uuid)'''
    ),
    (
        '''                metrics = VideoMetrics(video_id=video_id)''',
        '''                metrics = VideoMetrics(video_id=video_uuid)'''
    ),
    (
        '''            # Get all view sessions for this video
            stmt = select(ViewSession).where(ViewSession.video_id == video_id)''',
        '''            # Get all view sessions for this video
            stmt = select(ViewSession).where(ViewSession.video_id == video_uuid)'''
    ),
    (
        '''            # Get video for engagement metrics
            stmt = select(Video).where(Video.id == video_id)''',
        '''            # Get video for engagement metrics
            stmt = select(Video).where(Video.id == video_uuid)'''
    ),
    # get_video_metrics
    (
        '''    async def get_video_metrics(self, video_id: str) -> Optional[VideoMetrics]:
        """Get video metrics, calculating if necessary."""
        stmt = select(VideoMetrics).where(VideoMetrics.video_id == video_id)''',
        '''    async def get_video_metrics(self, video_id: str) -> Optional[VideoMetrics]:
        """Get video metrics, calculating if necessary."""
        import uuid as uuid_module
        video_uuid = uuid_module.UUID(video_id) if isinstance(video_id, str) else video_id
        
        stmt = select(VideoMetrics).where(VideoMetrics.video_id == video_uuid)'''
    ),
    # calculate_user_metrics
    (
        '''    async def calculate_user_metrics(self, user_id: str) -> UserBehaviorMetrics:
        """Calculate comprehensive metrics for a user."""
        try:
            # Check if metrics exist
            stmt = select(UserBehaviorMetrics).where(UserBehaviorMetrics.user_id == user_id)''',
        '''    async def calculate_user_metrics(self, user_id: str) -> UserBehaviorMetrics:
        """Calculate comprehensive metrics for a user."""
        try:
            import uuid as uuid_module
            user_uuid = uuid_module.UUID(user_id) if isinstance(user_id, str) else user_id
            
            # Check if metrics exist
            stmt = select(UserBehaviorMetrics).where(UserBehaviorMetrics.user_id == user_uuid)'''
    ),
    (
        '''                metrics = UserBehaviorMetrics(user_id=user_id)''',
        '''                metrics = UserBehaviorMetrics(user_id=user_uuid)'''
    ),
    (
        '''            # Get user
            stmt = select(User).where(User.id == user_id)''',
        '''            # Get user
            stmt = select(User).where(User.id == user_uuid)'''
    ),
    (
        '''            # Get view sessions for this user
            stmt = select(ViewSession).where(ViewSession.user_id == user_id)''',
        '''            # Get view sessions for this user
            stmt = select(ViewSession).where(ViewSession.user_id == user_uuid)'''
    ),
    (
        '''            # Get user's videos if they are a creator
            stmt = select(Video).where(Video.user_id == user_id)''',
        '''            # Get user's videos if they are a creator
            stmt = select(Video).where(Video.user_id == user_uuid)'''
    ),
    # get_user_metrics
    (
        '''    async def get_user_metrics(self, user_id: str) -> Optional[UserBehaviorMetrics]:
        """Get user metrics, calculating if necessary."""
        stmt = select(UserBehaviorMetrics).where(UserBehaviorMetrics.user_id == user_id)''',
        '''    async def get_user_metrics(self, user_id: str) -> Optional[UserBehaviorMetrics]:
        """Get user metrics, calculating if necessary."""
        import uuid as uuid_module
        user_uuid = uuid_module.UUID(user_id) if isinstance(user_id, str) else user_id
        
        stmt = select(UserBehaviorMetrics).where(UserBehaviorMetrics.user_id == user_uuid)'''
    ),
]

# Apply all replacements
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Replaced: {old[:60]}...")
    else:
        print(f"✗ Not found: {old[:60]}...")

# Write the file back
with open('app/analytics/services/enhanced_service.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ UUID conversions added successfully!")
