# scripts/setup/config_loader.py
import os
import yaml
import json
from typing import Dict, Any

class ConfigLoader:
    """
    Loads setup configs from YAML/JSON and applies environment overrides.

    Expected top-level keys sample:
      setup:
        host: 'dev'                  # or 'staging'/'prod'
        os_packages:
          - git
          - build-essential
        python:
          versions: ['3.10']
          venv_path: '/opt/socialflow/venv'
        node:
          install: true
          versions: ['18']
        docker:
          install: true
          registry: 'registry.example.com'
        kube:
          contexts:
            - name: 'dev-cluster'
              kubeconfig: '/etc/kube/dev.kubeconfig'
        users:
          - name: socialflow
            uid: 1500
            groups: ['docker']
        certificates:
          auto: false
          paths: {cert_dir: '/etc/ssl/socialflow'}
        secrets:
          bootstrap: false

    Environment variable overrides:
      SETUP__<SECTION>__<KEY>=value (double underscore denotes nesting)
    """
    def __init__(self, path: str = "setup.yaml"):
        self.path = path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        # Base load from file if present
        if os.path.exists(self.path):
            if self.path.endswith((".yaml", ".yml")):
                with open(self.path, "r") as fh:
                    self.config = yaml.safe_load(fh) or {}
            elif self.path.endswith(".json"):
                with open(self.path, "r") as fh:
                    self.config = json.load(fh)
            else:
                raise ValueError("Unsupported config format for setup file")
        else:
            self.config = {}

        # Ensure top-level 'setup' exists
        self.config.setdefault("setup", {})

        # Defaults
        s = self.config["setup"]
        s.setdefault("host", "dev")
        s.setdefault("os_packages", [])
        s.setdefault("python", {"versions": ["3.10"], "venv_path": "/opt/socialflow/venv"})
        s.setdefault("node", {"install": False, "versions": ["18"]})
        s.setdefault("docker", {"install": True, "registry": None})
        s.setdefault("kube", {"contexts": []})
        s.setdefault("users", [])
        s.setdefault("certificates", {"auto": False, "cert_dir": "/etc/ssl/socialflow"})
        s.setdefault("secrets", {"bootstrap": False})

        # Apply environment overrides prefixed with SETUP__
        for k, v in os.environ.items():
            if k.startswith("SETUP__"):
                # e.g. SETUP__python__venv_path -> ['python', 'venv_path']
                path = k[len("SETUP__"):].lower().split("__")
                d = self.config.setdefault("setup", {})
                for p in path[:-1]:
                    d = d.setdefault(p, {})
                # attempt JSON parse
                try:
                    parsed = json.loads(v)
                    d[path[-1]] = parsed
                except Exception:
                    d[path[-1]] = v

        return self.config
