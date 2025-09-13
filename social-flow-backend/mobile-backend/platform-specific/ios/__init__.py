# Package initializer for iOS module
"""
iOS platform-specific backend package.

Provides:
 - APNs push integration helpers (server-side)
 - IPA/delta update serving utilities
 - Device registry tuned for iOS clients (device model, iOS version, device token)
 - IPA signature & provisioning profile verification helpers
 - Analytics ingestion for iOS events
 - Universal Link (apple-app-site-association) validation helpers
 - Metrics aggregation and basic endpoints (app-level)

Author: Mobile Backend Team
"""

from .api import app

__all__ = ["app"]
