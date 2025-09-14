# Pydantic schema for structured logs
"""
Pydantic schema for structured logs.

- Use pydantic v1 or v2 (below written for v1/v2-compatible simple models).
- The schema is intentionally flexible but validates important fields like timestamp,
  service, level, and trace/span ids.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator

LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class StructuredLog(BaseModel):
    timestamp: datetime = Field(..., description="UTC timestamp of the event")
    service: str = Field(..., description="Logical service name")
    level: str = Field("INFO", description="Log level (DEBUG/INFO/...)")
    message: str = Field(..., description="Human readable message")
    logger: Optional[str] = Field(None, description="Logger name")
    host: Optional[str] = Field(None, description="Host or instance identifier")
    env: Optional[str] = Field(None, description="Environment (prod, staging, dev)")
    trace_id: Optional[str] = Field(None, description="Distributed trace id")
    span_id: Optional[str] = Field(None, description="Distributed span id")
    attrs: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary structured attributes")
    raw: Optional[str] = Field(None, description="Raw log line if available")

    @validator("level", pre=True, always=True)
    def normalize_level(cls, v):
        if v is None:
            return "INFO"
        s = str(v).upper()
        return s if s in LEVELS else "INFO"

    @validator("message")
    def message_length(cls, v):
        if len(v) > 100_000:
            # extremely defensive - reject giant messages
            raise ValueError("message too large")
        return v

    @root_validator(pre=True)
    def fill_timestamp(cls, values):
        # allow timestamp to be string or missing; if missing set now()
        ts = values.get("timestamp")
        if ts is None:
            values["timestamp"] = datetime.utcnow()
        return values

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }
