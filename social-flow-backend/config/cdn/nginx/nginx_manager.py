"""
# High-level manager (load configs, reload, test)
"""
# config/cdn/nginx/nginx_manager.py
"""
Nginx Manager
-------------
Handles lifecycle:
 - Build config
 - Write to filesystem
 - Validate syntax
 - Reload gracefully
"""

import subprocess
from typing import Optional


class NginxManager:
    def __init__(self, config_path: str = "/etc/nginx/nginx.conf"):
        self.config_path = config_path

    def write_config(self, content: str):
        """Write Nginx config to disk."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_config(self) -> bool:
        """Test Nginx config syntax."""
        result = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
        return result.returncode == 0

    def reload(self) -> bool:
        """Reload Nginx gracefully if config is valid."""
        if self.test_config():
            subprocess.run(["nginx", "-s", "reload"], check=True)
            return True
        return False

    def restart(self) -> bool:
        """Restart Nginx service (systemd)."""
        result = subprocess.run(["systemctl", "restart", "nginx"], capture_output=True)
        return result.returncode == 0
