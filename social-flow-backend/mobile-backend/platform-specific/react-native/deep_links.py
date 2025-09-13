# Deep link validation & resolution
"""
Deep link validation & normalization for React Native.

Checks:
 - Allowed schemes (http/https or custom app schemes)
 - Allowed domains for http/https
 - Normalizes to canonical form
"""

from urllib.parse import urlparse, urlunparse
from typing import Optional
from .config import CONFIG

ALLOWED_SCHEMES = {"http", "https", "myapp", "rnapp"}


class RNDeepLinkManager:
    def __init__(self):
        self.allowed_domains = set(CONFIG.allowed_deep_link_domains)

    def validate(self, url: str) -> bool:
        try:
            p = urlparse(url)
        except Exception:
            return False
        if p.scheme not in ALLOWED_SCHEMES:
            return False
        if p.scheme in ("http", "https"):
            domain = (p.netloc or "").split(":")[0].lower()
            if self.allowed_domains and domain not in self.allowed_domains:
                return False
        return True

    def normalize(self, url: str) -> Optional[str]:
        if not self.validate(url):
            return None
        p = urlparse(url)
        normalized = urlunparse((p.scheme, p.netloc, p.path or "/", p.params, p.query, ""))
        return normalized
