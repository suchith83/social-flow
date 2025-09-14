"""
Faker integration wrapper so the rest of the package can import a configured
faker instance without coupling to a specific faker library variant.
"""

from faker import Faker as _Faker
from typing import Any


def get_faker(locale: str = "en_US") -> _Faker:
    """
    Return a configured Faker instance for the desired locale.
    Wrapper exists so we can change seed strategy or pluggable providers later.
    """
    faker = _Faker(locale)
    # add providers or customizations here if needed
    return faker
