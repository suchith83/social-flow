# scripts/security/config_loader.py
import os
import yaml
import json
from typing import Dict, Any

class ConfigLoader:
    """
    Load security scan configuration from YAML/JSON and apply env overrides.
    Example config keys:
      security:
        dependency:
          enabled: true
          types: ['python', 'node']
        container:
          enabled: true
          scanner: 'trivy'
        static:
          enabled: true
          tools: ['bandit', 'eslint']
        secrets:
          enabled: true
          entropy_threshold: 4.5
        dynamic:
          enabled: false
          zap:
            host: 'localhost'
            port: 8080
        reporter:
          slack_webhook: null
    """
    def __init__(self, path: str = "security.yaml"):
        self.path = path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            if self.path.endswith((".yaml", ".yml")):
                with open(self.path, "r") as fh:
                    self.config = yaml.safe_load(fh) or {}
            elif self.path.endswith(".json"):
                with open(self.path, "r") as fh:
                    self.config = json.load(fh)
            else:
                raise ValueError("Unsupported config format")
        else:
            self.config = {}

        # defaults
        self.config.setdefault("security", {})
        s = self.config["security"]
        s.setdefault("dependency", {"enabled": True, "types": ["python"]})
        s.setdefault("container", {"enabled": True, "scanner": "trivy", "images": []})
        s.setdefault("static", {"enabled": True, "tools": ["bandit"]})
        s.setdefault("secrets", {"enabled": True, "entropy_threshold": 4.5, "paths": ["./"]})
        s.setdefault("dynamic", {"enabled": False})
        s.setdefault("reporter", {"slack_webhook": None, "output_dir": "./security-reports"})
        s.setdefault("ci", {"fail_on_high": True, "high_severity_threshold": 7})

        # apply env overrides starting with SEC_
        for k, v in os.environ.items():
            if k.startswith("SEC_"):
                path = k[len("SEC_"):].lower().split("__")
                d = self.config.setdefault("security", {})
                for p in path[:-1]:
                    d = d.setdefault(p, {})
                # try parse
                try:
                    parsed = json.loads(v)
                    d[path[-1]] = parsed
                except Exception:
                    d[path[-1]] = v

        return self.config
