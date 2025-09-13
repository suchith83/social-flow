# Package initializer for flutter module
"""
Flutter platform-specific backend package.

Provides:
 - Push integration (FCM / APNs wrappers)
 - Flutter bundle/AOT diffing and update serving
 - Device registry tuned for Flutter clients (engine/ABI)
 - Package/AOT signature verification helpers
 - Analytics ingestion for Flutter events
 - Deep link validation & normalization for Flutter URIs
 - Metrics aggregation

Author: Mobile Backend Team
"""

from .api import app

__all__ = ["app"]
