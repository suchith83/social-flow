"""
DataFactory: orchestrates generation, validation, serialization, and saving
of test fixtures used by unit and integration tests.
"""

import json
import os
from typing import Iterable, Any, Dict, List
from .config import DATA_CONFIG
from .generators import DataGenerators
from .serializers import Serializer
from .loaders import FixtureLoader
from .validators import validate_against_schema
from .utils import ensure_dir, atomic_write, fingerprint
from .exceptions import GeneratorError, FixtureLoadError


class DataFactory:
    """
    High-level facade for creating and persisting test fixtures.

    Example:
        factory = DataFactory(seed=123)
        users = factory.create_and_persist("users", factory.generators.batch(factory.generators.user, count=10))
    """

    def __init__(self, seed: int = None, config=None):
        self.config = config or DATA_CONFIG
        self.generators = DataGenerators(seed)
        self.serializer = Serializer()
        self.loader = FixtureLoader()
        ensure_dir(self.config.fixtures_dir)

    def create(self, kind: str, count: int = None, **kwargs) -> List[Dict[str, Any]]:
        """Create a list of items of 'kind' (e.g., 'users', 'products')."""
        if kind == "users":
            items = list(self.generators.batch(self.generators.user, count=count, **kwargs))
        elif kind == "products":
            items = list(self.generators.batch(self.generators.product, count=count, **kwargs))
        else:
            raise GeneratorError(f"Unknown kind: {kind}")
        return items

    def validate(self, kind: str, items: Iterable[Dict[str, Any]]):
        """Validate generated items against pydantic schemas."""
        for idx, item in enumerate(items):
            try:
                validate_against_schema(kind, item)
            except Exception as e:
                raise GeneratorError(f"Validation error for {kind}[{idx}]: {e}")

    def persist(self, kind: str, items: Iterable[Dict[str, Any]], filename: str = None) -> str:
        """
        Persist given items to fixtures directory. Returns path.
        Uses atomic write.
        """
        if filename is None:
            filename = f"{kind}.json"
        path = os.path.join(self.config.fixtures_dir, filename)
        data = self.serializer.dumps(list(items))
        atomic_write(path, data)
        return path

    def create_and_persist(self, kind: str, count: int = None, filename: str = None, **kwargs) -> str:
        """Create, validate and persist items. Returns path to fixture file."""
        items = self.create(kind, count=count, **kwargs)
        self.validate(kind, items)
        return self.persist(kind, items, filename=filename)

    def load(self, path: str):
        """Load a fixture file and return deserialized objects."""
        try:
            return self.loader.load(path)
        except Exception as e:
            raise FixtureLoadError(f"Failed to load fixture {path}: {e}")

    def fingerprint(self, items: Iterable[Dict[str, Any]]) -> str:
        return fingerprint(list(items))
