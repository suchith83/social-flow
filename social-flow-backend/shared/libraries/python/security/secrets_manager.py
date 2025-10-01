# secrets_manager.py
import os
import json
import requests


class SecretsManager:
    """
    Secrets manager for secure retrieval of secrets.
    Supports:
    - Environment variables
    - HashiCorp Vault
    - AWS Secrets Manager
    """

    def __init__(self, vault_url=None, aws_url=None, token=None):
        self.vault_url = vault_url
        self.aws_url = aws_url
        self.token = token

    def from_env(self, key: str) -> str:
        return os.getenv(key)

    def from_vault(self, path: str) -> dict:
        if not self.vault_url:
            return {}
        headers = {"X-Vault-Token": self.token}
        resp = requests.get(f"{self.vault_url}/v1/{path}", headers=headers)
        return resp.json().get("data", {})

    def from_aws(self, secret_name: str) -> dict:
        if not self.aws_url:
            return {}
        resp = requests.get(f"{self.aws_url}/{secret_name}", headers={"Authorization": f"Bearer {self.token}"})
        return json.loads(resp.text)
