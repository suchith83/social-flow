"""
pytest fixtures and helpers for tests. Provide parametrized fixtures that
use DataFactory under the hood to produce deterministic test data.
"""

import pytest
from .data_factory import DataFactory
from .config import DATA_CONFIG


@pytest.fixture(scope="session")
def data_factory():
    """Session-scoped DataFactory with default seed from config."""
    return DataFactory(seed=DATA_CONFIG.default_seed)


@pytest.fixture
def user_data(data_factory):
    """Function-scoped fixture returning a factory function to get users."""
    def _make(count: int = 5, **overrides):
        return data_factory.create("users", count=count, **overrides)
    return _make


@pytest.fixture
def persisted_users(tmp_path, data_factory):
    """Create and persist a small set of users to a temp fixtures dir (isolated)."""
    # override factory config to persist into tmp_path
    data_factory.config = data_factory.config  # keep existing; we will pass filename
    path = data_factory.create_and_persist("users", count=3, filename="persisted_users.json")
    # load back
    return data_factory.load(path)
