"""
Schema validation and convenience validators.
Wraps pydantic models and converts errors into package exceptions.
"""

from typing import Any
from .schema import User, Product
from .exceptions import SchemaValidationError


def validate_against_schema(kind: str, obj: Any) -> None:
    """
    Validate object against known schema types. Raises SchemaValidationError on failure.
    """
    try:
        if kind == "users":
            # pydantic will coerce and validate types
            User.parse_obj(obj)
        elif kind == "products":
            Product.parse_obj(obj)
        else:
            # If unknown schema, no-op (but could raise if strict mode on)
            return
    except Exception as exc:
        # Wrap in explicit schema error for caller clarity
        raise SchemaValidationError(str(exc))
