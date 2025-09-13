"""
# Configuration models & schema validation
"""
# config/cdn/cloudflare/cf_config.py
"""
Cloudflare Config
-----------------
Reusable schemas for zone/DNS configuration.
"""

from typing import Dict


class CloudflareConfig:
    @staticmethod
    def dns_record(name: str, record_type: str, content: str, proxied: bool = True) -> Dict:
        """Build DNS record payload."""
        return {
            "type": record_type,
            "name": name,
            "content": content,
            "ttl": 120,
            "proxied": proxied,
        }

    @staticmethod
    def page_rule(url_pattern: str, settings: Dict) -> Dict:
        """Build page rule configuration."""
        return {"targets": [{"target": "url", "constraint": {"operator": "matches", "value": url_pattern}}],
                "actions": [{"id": k, "value": v} for k, v in settings.items()]}
