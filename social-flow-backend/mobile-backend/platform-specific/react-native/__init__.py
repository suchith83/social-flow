# Package initializer for react-native module
"""
React Native platform-specific backend package.

Exports FastAPI app (api.app). Provides:
 - Device registry tailored to RN clients
 - Push (FCM / APNs) glue
 - JS bundle (OTA) diffing & serving (CodePush-like)
 - Signature verification scaffolding
 - Analytics ingestion tailored to RN events
 - Deep link normalization & validation
 - Metrics collection

Author: Mobile Backend Team
"""

from .api import app

__all__ = ["app"]
