# Validation wrapper around schema
"""
Validation and normalization helpers that wrap the Pydantic schema.
Exposes a validate() function that returns a StructuredLog object or raises.
"""

from typing import Tuple, Optional
from .schema import StructuredLog
from pydantic import ValidationError


def validate(payload: dict) -> Tuple[Optional[StructuredLog], Optional[ValidationError]]:
    """
    Validate and coerce payload into StructuredLog model.

    Returns tuple (model, error). If invalid, model is None and error contains
    the ValidationError instance.
    """
    try:
        model = StructuredLog.parse_obj(payload)
        return model, None
    except ValidationError as e:
        return None, e
