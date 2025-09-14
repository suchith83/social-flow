# scripts/deployment/config_loader.py
import os
import yaml
import json
from typing import Dict, Any


class ConfigLoader:
    """
    Loads deployment configuration from YAML, JSON, or environment variables.
    Provides validation and environment overrides for flexible deployments.
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if not self.config_path:
            raise ValueError("Config path not provided")

        if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif self.config_path.endswith(".json"):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError("Unsupported config format. Use YAML or JSON.")

        self._apply_env_overrides()
        self._validate()
        return self.config

    def _apply_env_overrides(self):
        for key, value in os.environ.items():
            if key.startswith("DEPLOY_"):
                path = key.replace("DEPLOY_", "").lower().split("__")
                d = self.config
                for p in path[:-1]:
                    d = d.setdefault(p, {})
                d[path[-1]] = value

    def _validate(self):
        required_keys = ["app", "docker", "k8s", "infra"]
        for k in required_keys:
            if k not in self.config:
                raise KeyError(f"Missing required config key: {k}")


if __name__ == "__main__":
    loader = ConfigLoader("deployment.yaml")
    print(loader.load())
