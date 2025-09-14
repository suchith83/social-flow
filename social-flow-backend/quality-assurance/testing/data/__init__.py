"""
quality-assurance.testing.data

Utilities for generating, validating, serializing, and loading test data.
Designed for deterministic and random test-data generation suitable for
unit/integration tests and CI fixture generation.
"""

__version__ = "1.0.0"

from .config import DATA_CONFIG  # re-export config
from .generators import DataGenerators
from .data_factory import DataFactory
from .fixtures import pytest_fixtures  # convenience import for tests
