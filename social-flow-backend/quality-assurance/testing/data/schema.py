"""
Pydantic schemas for test data objects.
Define domain-agnostic example schemas that can be used/extended.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, conint, constr
from datetime import datetime


class Address(BaseModel):
    street: constr(strip_whitespace=True, min_length=1, max_length=128)
    city: constr(strip_whitespace=True, min_length=1, max_length=64)
    state: Optional[constr(strip_whitespace=True, max_length=64)]
    postal_code: constr(strip_whitespace=True, min_length=3, max_length=16)
    country: constr(strip_whitespace=True, min_length=2, max_length=64)


class User(BaseModel):
    id: int = Field(..., ge=1)
    username: constr(strip_whitespace=True, min_length=3, max_length=64)
    email: EmailStr
    active: bool = True
    created_at: datetime
    roles: List[constr(min_length=1, max_length=32)] = []
    address: Optional[Address] = None


class Product(BaseModel):
    sku: constr(strip_whitespace=True, min_length=1, max_length=64)
    name: constr(strip_whitespace=True, min_length=1, max_length=256)
    price_cents: conint(ge=0) = 0
    in_stock: bool = True
    tags: List[constr(min_length=1, max_length=32)] = []


# Add other domain schemas as needed.
