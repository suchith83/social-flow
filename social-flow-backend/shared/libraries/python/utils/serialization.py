# serialization.py
import json
import yaml
import pickle
from typing import Any


class Serializer:
    """
    Serialization utilities for JSON, YAML, and Pickle.
    """

    @staticmethod
    def to_json(data: Any) -> str:
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(data: str) -> Any:
        return json.loads(data)

    @staticmethod
    def to_yaml(data: Any) -> str:
        return yaml.safe_dump(data)

    @staticmethod
    def from_yaml(data: str) -> Any:
        return yaml.safe_load(data)

    @staticmethod
    def to_pickle(data: Any) -> bytes:
        return pickle.dumps(data)

    @staticmethod
    def from_pickle(data: bytes) -> Any:
        return pickle.loads(data)
