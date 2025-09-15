from pydantic import BaseModel, Field
from datetime import datetime


class Event(BaseModel):
    """Schema for validating incoming events."""

    event_id: str = Field(..., description="Unique event identifier")
    user_id: str = Field(..., description="User identifier")
    event_type: str = Field(..., description="Type of event (click, view, etc.)")
    value: float = Field(..., description="Numeric value associated with the event")
    metadata: dict = Field(default_factory=dict, description="Additional attributes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
