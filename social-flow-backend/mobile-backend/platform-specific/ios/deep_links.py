# Deep link validation & resolution
"""
Universal Link and deep link validation for iOS.

Responsibilities:
 - Validate potential Universal Links and app links
 - Ensure allowed domains (apple-app-site-association endpoints) or configured domain list
 - Normalize links and prevent unsafe schemes/open-redirects

Note:
 - For Universal Links, server should host apple-app-site-association JSON at /.well-known/apple-app-site-association
   and the app must have matching entitlements. This module provides validation helpers, not the AASA host.
"""

from urllib.parse import urlparse, urlunparse
from typing import Optional
from .config import CONFIG


class UniversalLinkManager:
    def __init__(self):
        self.allowed_domains = set(CONFIG.allowed_universal_link_domains)

    def validate(self, url: str) -> bool:
        """Return True if deep link is allowed and safe."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in ("http", "https", "app", "myapp"):
            return False
        if parsed.scheme in ("http", "https"):
            domain = parsed.netloc.split(":")[0].lower()
            if self.allowed_domains and domain not in self.allowed_domains:
                return False
        return True

    def normalize(self, url: str) -> Optional[str]:
        if not self.validate(url):
            return None
        p = urlparse(url)
        normalized = urlunparse((p.scheme, p.netloc, p.path or "/", p.params, p.query, ""))
        return normalized
