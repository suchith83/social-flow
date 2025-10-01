# common/libraries/python/messaging/serializers.py
"""
Serialization utilities for messaging.
"""

import json
from typing import Any, Dict

class JSONSerializer:
    @staticmethod
    def dumps(message: Dict[str, Any]) -> bytes:
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def loads(data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))

# Placeholder for Protobuf serializer
class ProtobufSerializer:
    @staticmethod
    def dumps(message: Any) -> bytes:
        raise NotImplementedError("Protobuf serialization not implemented yet.")

    @staticmethod
    def loads(data: bytes) -> Any:
        raise NotImplementedError("Protobuf deserialization not implemented yet.")

SERIALIZERS = {
    "json": JSONSerializer,
    "protobuf": ProtobufSerializer,
}
