# config.py
import os
import json
import yaml
from dotenv import load_dotenv


class Config:
    """
    Loads configuration from JSON, YAML, .env, or environment variables.
    """

    def __init__(self, filepath: str = None):
        self.config = {}
        if filepath:
            self.load(filepath)
        else:
            load_dotenv()

    def load(self, filepath: str):
        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                self.config = json.load(f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r") as f:
                self.config = yaml.safe_load(f)
        elif filepath.endswith(".env"):
            load_dotenv(filepath)
            self.config = dict(os.environ)
        else:
            raise ValueError("Unsupported config format")

    def get(self, key: str, default=None):
        return self.config.get(key) or os.getenv(key, default)
