"""
Follow model - Re-export from consolidated models.

DEPRECATED: Import from app.models.social instead.
This file exists only for backward compatibility.
"""

# Re-export from consolidated models
from app.models.social import Follow

__all__ = ["Follow"]
