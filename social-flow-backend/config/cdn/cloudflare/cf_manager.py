"""
# High-level orchestration of zones, DNS, firewall, cache
"""
# config/cdn/cloudflare/cf_manager.py
"""
Cloudflare Manager
------------------
High-level orchestration of Cloudflare zones, DNS, firewall, cache.
"""

import requests
from typing import Dict, Any


class CloudflareManager:
    def __init__(self, api_token: str):
        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    def list_zones(self) -> Dict[str, Any]:
        """List all Cloudflare zones."""
        resp = requests.get(f"{self.base_url}/zones", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_zone_id(self, domain: str) -> str:
        """Fetch Cloudflare zone ID for a domain."""
        zones = self.list_zones()["result"]
        for z in zones:
            if z["name"] == domain:
                return z["id"]
        raise ValueError(f"Zone not found: {domain}")

    def create_dns_record(self, zone_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create DNS record in Cloudflare."""
        resp = requests.post(f"{self.base_url}/zones/{zone_id}/dns_records",
                             headers=self.headers, json=record)
        resp.raise_for_status()
        return resp.json()

    def purge_cache(self, zone_id: str, files: list) -> Dict[str, Any]:
        """Purge specific files from cache."""
        payload = {"files": files}
        resp = requests.post(f"{self.base_url}/zones/{zone_id}/purge_cache",
                             headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()
