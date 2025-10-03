"""
Video model - Re-export from consolidated models.

DEPRECATED: Import from app.models.video instead.
This file exists only for backward compatibility.
"""

# Re-export from consolidated models
from app.models.video import Video, VideoStatus, VideoVisibility

__all__ = ["Video", "VideoStatus", "VideoVisibility"]
