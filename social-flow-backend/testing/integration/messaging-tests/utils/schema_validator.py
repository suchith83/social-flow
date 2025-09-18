"""
Schema validation helpers (JSON Schema + optional Avro/fastavro).
Tests should provide expected schemas and call these helpers.
"""

import json
from jsonschema import validate as jsonschema_validate, ValidationError
from typing import Any, Dict, Union
from fastavro.validation import validate as avro_validate  # optional

def validate_json_schema(instance: Union[dict, list], schema: Dict[str, Any]) -> None:
    """
    Raises jsonschema.ValidationError if invalid.
    """
    jsonschema_validate(instance=instance, schema=schema)

def validate_avro(instance: dict, avro_schema: Dict[str, Any]) -> bool:
    """
    Returns True if instance valid according to avro_schema; raises on errors.
    """
    # fastavro's validate returns bool
    return avro_validate(instance, avro_schema)
