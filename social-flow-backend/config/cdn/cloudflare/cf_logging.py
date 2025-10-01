"""
# Access logs, analytics, monitoring
"""
# config/cdn/cloudflare/cf_logging.py
"""
Cloudflare Logging
------------------
Fetch analytics & access logs.
"""

import requests
from typing import Dict, Any


class CloudflareLogging:
    def __init__(self, api_token: str):
        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    def analytics(self, zone_id: str) -> Dict[str, Any]:
        """Fetch traffic analytics for zone."""
        resp = requests.get(f"{self.base_url}/zones/{zone_id}/analytics/dashboard", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def firewall_events(self, zone_id: str) -> Dict[str, Any]:
        """Fetch firewall event logs."""
        resp = requests.get(f"{self.base_url}/zones/{zone_id}/security/events", headers=self.headers)
        resp.raise_for_status()
        return resp.json()
