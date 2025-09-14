"""
Generators for randomized and deterministic test data.
Wraps faker and python stdlib to provide easy-to-use factories.
"""

from typing import Any, Dict, Iterable, List, Optional
import random
import time
from datetime import datetime, timezone
from .config import DATA_CONFIG
from .faker_integration import get_faker
from .schema import User, Address, Product
from .exceptions import GeneratorError
from .utils import fingerprint

Faker = get_faker(DATA_CONFIG.locale)


class DataGenerators:
    """High-level data generators usable by tests."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else DATA_CONFIG.default_seed
        self.random = random.Random(self.seed)
        self.faker = Faker
        # Make faker deterministic for the chosen seed
        try:
            self.faker.seed_instance(self.seed)
        except Exception:
            # Some faker versions use seed; ignore if not available
            self.random.seed(self.seed)

    def timestamp_now(self) -> datetime:
        """Return a timezone-aware 'now' for fixtures."""
        return datetime.now(timezone.utc)

    def user(self, uid: Optional[int] = None, **overrides) -> Dict[str, Any]:
        """
        Create a realistic user dict that matches schema.User.
        Accepts overrides for any field.
        """
        try:
            id_val = uid if uid is not None else self.random.randint(1, 10_000_000)
            created = overrides.get("created_at", self.timestamp_now())
            user = {
                "id": id_val,
                "username": overrides.get("username", self.faker.user_name()[:64]),
                "email": overrides.get("email", self.faker.safe_email()),
                "active": overrides.get("active", True),
                "created_at": created,
                "roles": overrides.get("roles", ["user"]),
            }

            # Optionally include address
            if overrides.get("address", True) is not False:
                addr = overrides.get("address")
                if addr is None:
                    addr = {
                        "street": self.faker.street_address(),
                        "city": self.faker.city(),
                        "state": self.faker.state(),
                        "postal_code": self.faker.postcode(),
                        "country": self.faker.current_country() if hasattr(self.faker, "current_country") else self.faker.country()
                    }
                user["address"] = addr

            return user
        except Exception as e:
            raise GeneratorError(f"Failed to generate user: {e}")

    def product(self, sku: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """Create a product dict that matches schema.Product."""
        try:
            sku_val = sku or f"SKU-{self.random.randint(100000, 999999)}"
            product = {
                "sku": sku_val,
                "name": overrides.get("name", self.faker.sentence(nb_words=3)[:256]),
                "price_cents": overrides.get("price_cents", self.random.randint(0, 100_000)),
                "in_stock": overrides.get("in_stock", True),
                "tags": overrides.get("tags", [self.faker.word()[:32] for _ in range(self.random.randint(0, 3))])
            }
            return product
        except Exception as e:
            raise GeneratorError(f"Failed to generate product: {e}")

    def batch(self, generator_fn, count: int = None, **kwargs) -> Iterable[Dict[str, Any]]:
        """Yield 'count' items from generator_fn. Deterministic for a seed."""
        count = count or DATA_CONFIG.default_batch_size
        for i in range(count):
            kwargs_with_index = dict(kwargs)
            # allow index-aware factories
            if "index" in generator_fn.__code__.co_varnames:
                kwargs_with_index["index"] = i
            yield generator_fn(i if "uid" in generator_fn.__code__.co_varnames else None, **kwargs_with_index) \
                if generator_fn.__name__ == "user" else generator_fn(None if generator_fn.__name__ != "product" else None, **kwargs_with_index)

    def fingerprint_for(self, obj: Any) -> str:
        """Return a fingerprint for generated object set (convenience)."""
        return fingerprint(obj)
