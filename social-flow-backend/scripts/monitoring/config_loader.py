# scripts/monitoring/config_loader.py
import os
import yaml
import json
from typing import Dict, Any


class ConfigLoader:
    """
    Loads monitoring configuration from YAML/JSON and applies env overrides.
    Environment overrides are prefixed with MON_. Example:
      MON_synthetic__interval=60
    will set config['synthetic']['interval'] = '60'
    """

    def __init__(self, path: str = "monitoring.yaml"):
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
                raise ValueError("Unsupported config format. Use .yaml/.yml or .json")
        else:
            self.config = {}

        # Provide defaults
        self.config.setdefault("monitoring", {})
        m = self.config["monitoring"]
        m.setdefault("prometheus", {"enabled": True, "port": 9101})
        m.setdefault("metrics", {"collection_interval": 15})
        m.setdefault("synthetic", {"checks": [], "interval": 60})
        m.setdefault("logs", {"paths": [], "patterns": []})
        m.setdefault("alerts", {"slack_webhook": None, "pagerduty_key": None, "email": {}})
        m.setdefault("dashboard", {"output_dir": "./dashboards"})

        # Apply env overrides
        for k, v in os.environ.items():
            if k.startswith("MON_"):
                path = k[len("MON_"):].lower().split("__")
                d = self.config.setdefault("monitoring", {})
                for p in path[:-1]:
                    d = d.setdefault(p, {})
                # attempt to parse json-ish env var values
                try:
                    parsed = json.loads(v)
                    d[path[-1]] = parsed
                except Exception:
                    d[path[-1]] = v

        return self.config
