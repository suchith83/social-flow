# scripts/maintenance/config_loader.py
import os
import yaml
import json
from typing import Dict, Any

class ConfigLoader:
    """
    Loads maintenance configuration from YAML / JSON or environment overrides.

    Example keys expected in config:
      maintenance:
        backup:
          enabled: true
          s3_bucket: my-backups
          retention_days: 30
        db:
          type: postgres
          dsn: ...
          vacuum: true
          vacuum_full: false
        logs:
          rotate_after_mb: 50
          retention_days: 14
        cleanup:
          temp_paths: ['/tmp/socialflow']
          older_than_days: 7
        notifications:
          slack_webhook: ...
    """

    def __init__(self, path: str = "maintenance.yaml"):
        self.path = path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            if self.path.endswith((".yaml", ".yml")):
                with open(self.path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
            elif self.path.endswith(".json"):
                with open(self.path, "r") as f:
                    self.config = json.load(f)
            else:
                raise ValueError("Unsupported config format, use .yaml/.yml or .json")
        else:
            self.config = {}

        # apply environment overrides prefixed with MAINT_
        for k, v in os.environ.items():
            if k.startswith("MAINT_"):
                # e.g. MAINT_backup__s3_bucket => {'backup': {'s3_bucket': v}}
                path = k[len("MAINT_"):].lower().split("__")
                d = self.config.setdefault("maintenance", {})
                for p in path[:-1]:
                    d = d.setdefault(p, {})
                d[path[-1]] = v

        # provide defaults
        self.config.setdefault("maintenance", {})
        m = self.config["maintenance"]
        m.setdefault("backup", {"enabled": False, "retention_days": 30})
        m.setdefault("db", {"vacuum": True, "vacuum_full": False})
        m.setdefault("logs", {"rotate_after_mb": 50, "retention_days": 14})
        m.setdefault("cleanup", {"temp_paths": [], "older_than_days": 7})
        m.setdefault("notifications", {})

        return self.config
