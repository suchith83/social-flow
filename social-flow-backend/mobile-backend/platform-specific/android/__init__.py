# Package initializer for android module
"""
Android platform-specific backend package.

Contains APIs and helpers for:
 - FCM push management
 - APK/delta update serving
 - Device registry and capabilities
 - APK signature verification
 - Android-specific analytics ingestion
 - Deep link validation
 - Metrics for android flows

Author: Mobile Backend Team
"""

from .api import app  # FastAPI app

__all__ = ["app"]
