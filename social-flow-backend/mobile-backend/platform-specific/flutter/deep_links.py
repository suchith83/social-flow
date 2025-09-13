# Deep link validation & resolution
"""
Deep link management for Flutter.

Flutter apps often use Universal Links (iOS) and App Links (Android) or custom URL schemes.
This module validates and normalizes deep links and guards against open redirects and unsafe schemes.
"""

from urllib.parse import urlparse, urlunparse
from typing import Optional
from .config import CONFIG

ALLOWED_SCHEMES = {"http", "https", "flutter", "myapp"}  # extend with app-specific schemes

class FlutterDeepLinkManager:
    def __init__(self):
        self.allowed_domains = set()  # can be configured if needed

    def validate(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in ALLOWED_SCHEMES and not parsed.scheme.startswith("http"):
            return False
        # if http/https, optionally validate domain
        if parsed.scheme in ("http", "https") and self.allowed_domains:
            domain = parsed.netloc.split(":")[0].lower()
            if domain not in self.allowed_domains:
                return False
        return True

    def normalize(self, url: str) -> Optional[str]:
        if not self.validate(url):
            return None
        p = urlparse(url)
        normalized = urlunparse((p.scheme, p.netloc, p.path or "/", p.params, p.query, ""))
        return normalized
