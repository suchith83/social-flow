"""
# Common helpers (zone validation, request signing)
"""
# config/cdn/cloudflare/cf_utils.py
"""
Cloudflare Utilities
--------------------
Helpers for validation & ID management.
"""

import re
import uuid


class CloudflareUtils:
    @staticmethod
    def generate_request_id() -> str:
        """Generate unique ID for API request tracking."""
        return str(uuid.uuid4())

    @staticmethod
    def validate_domain(domain: str) -> bool:
        """Validate domain format."""
        return re.match(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z]{2,6})+$", domain) is not None

    @staticmethod
    def pretty_json(data: dict) -> str:
        """Return JSON in pretty-printed format."""
        import json
        return json.dumps(data, indent=2)
