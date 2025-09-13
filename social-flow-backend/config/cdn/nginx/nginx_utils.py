"""
# Common helpers (validation, process control)
"""
# config/cdn/nginx/nginx_utils.py
"""
Nginx Utils
-----------
Helpers for validation and service control.
"""

import subprocess
import re


class NginxUtils:
    @staticmethod
    def validate_domain(domain: str) -> bool:
        """Validate domain name format."""
        return re.match(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z]{2,6})+$", domain) is not None

    @staticmethod
    def version() -> str:
        """Get Nginx version."""
        result = subprocess.run(["nginx", "-v"], capture_output=True, text=True)
        return result.stderr.strip() if result.returncode == 0 else "Unknown"

    @staticmethod
    def test_binary() -> bool:
        """Check if nginx binary is available."""
        result = subprocess.run(["which", "nginx"], capture_output=True)
        return result.returncode == 0
