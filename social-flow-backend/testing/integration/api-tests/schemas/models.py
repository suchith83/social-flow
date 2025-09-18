"""
Pydantic models used for contract validation in tests.

Adjust fields to your API contracts. These models allow strict response validation,
and make contract tests readable and maintainable.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class AuthToken(BaseModel):
    token: str
    expires_in: Optional[int] = Field(None, description="seconds until expiry")


class UserModel(BaseModel):
    id: Optional[str]
    username: str
    email: Optional[str]
    role: Optional[str]


class StorageObject(BaseModel):
    key: str
    size: int
    content_type: Optional[str]


class BucketList(BaseModel):
    buckets: List[str]
