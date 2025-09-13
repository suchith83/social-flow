"""
# Cache rules, purge, page rules
"""
# config/cdn/cloudflare/cf_cache.py
"""
Cloudflare Cache
----------------
Manages cache purge, rules, and custom settings.
"""

import requests
from typing import Dict, Any


class CloudflareCache:
    def __init__(self, api_token: str):
        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    def purge_all(self, zone_id: str) -> Dict[str, Any]:
        """Purge all cache."""
        resp = requests.post(f"{self.base_url}/zones/{zone_id}/purge_cache",
                             headers=self.headers, json={"purge_everything": True})
        resp.raise_for_status()
        return resp.json()

    def cache_settings(self, zone_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cache settings."""
        resp = requests.patch(f"{self.base_url}/zones/{zone_id}/settings",
                              headers=self.headers, json=settings)
        resp.raise_for_status()
        return resp.json()
