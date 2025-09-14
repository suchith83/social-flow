# scripts/security/secret_store.py
import os
import logging
from typing import Optional

logger = logging.getLogger("security.secret_store")

class SecretStore:
    """
    Small abstraction to read secrets from environment, AWS Secrets Manager, or files.
    Designed to avoid accidental logging of secret values.
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("security", {}).get("secret_backend", "env")

    def get(self, key: str) -> Optional[str]:
        """
        Return secret value or None. Avoids printing values to logs.
        """
        if self.backend == "env":
            return os.environ.get(key)
        elif self.backend == "file":
            path = os.environ.get(key + "_FILE")  # conventional pattern
            if path and os.path.exists(path):
                with open(path, "r") as fh:
                    return fh.read().strip()
            return None
        # Add AWS Secrets Manager, Vault, etc. as needed, using SDKs (not included to avoid creds here)
        else:
            logger.warning("Unknown secret backend: %s", self.backend)
            return None
