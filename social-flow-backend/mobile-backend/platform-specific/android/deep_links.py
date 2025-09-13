# Deep link validation & resolution
"""
Deep link validation and resolution.

 - Validate incoming deep links against allowed domains
 - Normalize links and optionally resolve to in-app routing
 - Protect against open-redirects and unsafe schemes
"""

from urllib.parse import urlparse, urlunparse
from typing import Optional
from .config import CONFIG


class DeepLinkManager:
    def __init__(self):
        self.allowed_domains = set(CONFIG.allowed_deep_link_domains)

    def validate(self, url: str) -> bool:
        """Return True if deep link is allowed and safe."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        # require http(s) schemes or android-app: for app links
        if parsed.scheme not in ("http", "https", "android-app"):
            return False
        # domain checks for http/https
        if parsed.scheme in ("http", "https"):
            domain = parsed.netloc.split(":")[0].lower()
            if domain not in self.allowed_domains:
                return False
        # prevent data: or javascript: etc by scheme check above
        return True

    def normalize(self, url: str) -> Optional[str]:
        """Normalize URL and return canonical string or None if invalid."""
        if not self.validate(url):
            return None
        p = urlparse(url)
        # strip fragments, normalize path
        normalized = urlunparse((p.scheme, p.netloc, p.path or "/", p.params, p.query, ""))
        return normalized
