# scripts/setup/secrets_bootstrap.py
import logging
import os
from typing import Dict, Any
from .utils import write_file, ensure_dir

logger = logging.getLogger("setup.secrets")

class SecretsBootstrap:
    """
    Lightweight secrets bootstrapping to Secret Manager placeholders.
    WARNING: This module does not store secrets in plain text in repo. It provides helpers to:
      - Create a secrets manifest template
      - Upload a secret from environment or files to a secrets manager (abstract)
      - Mark secrets as bootstrapped via sentinel files
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {}).get("secrets", {})
        self.output_dir = self.cfg.get("output_dir", "/var/lib/socialflow/secrets")
        ensure_dir(self.output_dir)

    def generate_template(self, keys: Dict[str, str], path: str = None):
        """
        Write a template for secret keys (no values) to be filled by operators.
        """
        path = path or os.path.join(self.output_dir, "secrets-template.yaml")
        content = "\n".join(f"{k}: \"REPLACE_ME\"" for k in keys.keys())
        write_file(path, content, mode=0o600)
        logger.info("Wrote secrets template to %s", path)
        return path

    def bootstrap_from_env(self, mapping: Dict[str, str], sentinel: str = None):
        """
        For each mapping entry, look for value in environment and store into simple file (as placeholder).
        Real deployments should call secret manager APIs here.
        """
        sentinel = sentinel or os.path.join(self.output_dir, ".bootstrapped")
        created = []
        for key, env_name in mapping.items():
            val = os.environ.get(env_name)
            if not val:
                logger.warning("No env var %s provided for secret key %s; skipping", env_name, key)
                continue
            dest = os.path.join(self.output_dir, key)
            write_file(dest, val, mode=0o600)
            created.append(dest)
        # create sentinel
        write_file(sentinel, "bootstrapped\n", mode=0o600)
        logger.info("Bootstrapped %d secrets to %s", len(created), self.output_dir)
        return created
