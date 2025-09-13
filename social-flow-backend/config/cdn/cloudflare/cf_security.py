"""
# Security (firewall rules, WAF, DDoS, bot protection)
"""
# config/cdn/cloudflare/cf_security.py
"""
Cloudflare Security
-------------------
Handles:
 - Firewall rules
 - WAF configs
 - Rate limiting
 - Bot protection
"""

import requests
from typing import Dict, Any


class CloudflareSecurity:
    def __init__(self, api_token: str):
        self.base_url = "https://api.cloudflare.com/client/v4"
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    def add_firewall_rule(self, zone_id: str, description: str, expression: str, action: str = "block") -> Dict[str, Any]:
        """Add firewall rule."""
        payload = {"filter": {"expression": expression}, "action": action, "description": description}
        resp = requests.post(f"{self.base_url}/zones/{zone_id}/firewall/rules",
                             headers=self.headers, json=[payload])
        resp.raise_for_status()
        return resp.json()

    def enable_waf(self, zone_id: str, package_id: str) -> Dict[str, Any]:
        """Enable WAF package for zone."""
        resp = requests.patch(f"{self.base_url}/zones/{zone_id}/firewall/waf/packages/{package_id}",
                              headers=self.headers, json={"sensitivity": "high", "action_mode": "challenge"})
        resp.raise_for_status()
        return resp.json()

    def configure_rate_limit(self, zone_id: str, url: str, threshold: int, period: int) -> Dict[str, Any]:
        """Apply rate limit rule."""
        payload = {
            "threshold": threshold,
            "period": period,
            "action": {"mode": "challenge"},
            "match": {"request": {"url": url}},
        }
        resp = requests.post(f"{self.base_url}/zones/{zone_id}/rate_limits",
                             headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()
