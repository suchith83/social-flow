# scripts/setup/env_manager.py
import os
import logging
from typing import Dict, Any
from .utils import write_file, ensure_dir

logger = logging.getLogger("setup.env_manager")

class EnvManager:
    """
    Manage environment variables and .env files for services.

    Features:
      - Render .env from config dict
      - Merge existing .env safely (preserve keys unless override=True)
      - Optionally create systemd environment files
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {}).get("env", {})
        self.default_dir = self.cfg.get("dir", "/etc/socialflow/env")
        self.systemd_dir = "/etc/systemd/system"

    def render_dotenv(self, mapping: Dict[str, str], path: str, override: bool = False):
        """
        Writes a simple KEY=VALUE file. If override=False, merges with existing.
        """
        ensure_dir(os.path.dirname(path) or ".")
        existing = {}
        if os.path.exists(path) and not override:
            with open(path, "r") as fh:
                for ln in fh:
                    if "=" in ln and not ln.strip().startswith("#"):
                        k, v = ln.strip().split("=", 1)
                        existing[k] = v
        # merge (existing keys persist unless override)
        new_map = existing.copy()
        new_map.update(mapping)
        lines = [f'{k}="{v}"' if " " in v else f"{k}={v}" for k, v in new_map.items()]
        write_file(path, "\n".join(lines) + "\n")
        logger.info("Wrote .env file to %s", path)

    def write_systemd_env(self, service_name: str, mapping: Dict[str, str]):
        """
        Writes an EnvironmentFile for systemd unit to include.
        """
        unit_env_path = f"/etc/systemd/system/{service_name}.env"
        self.render_dotenv(mapping, unit_env_path, override=True)
        logger.info("Wrote systemd environment file %s", unit_env_path)
        return unit_env_path
