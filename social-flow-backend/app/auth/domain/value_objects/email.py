"""
Email Value Object

Represents and validates email addresses in the domain.
"""

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Email:
    """
    Email value object with validation.
    
    Immutable value object representing an email address.
    Ensures email format validity at construction time.
    """
    
    value: str
    
    # Email validation regex pattern
    EMAIL_REGEX: ClassVar[re.Pattern] = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def __post_init__(self) -> None:
        """Validate email format on creation."""
        if not self.value:
            raise ValueError("Email cannot be empty")
        
        if len(self.value) > 255:
            raise ValueError("Email cannot exceed 255 characters")
        
        if not self.EMAIL_REGEX.match(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
        
        # Normalize email to lowercase
        object.__setattr__(self, 'value', self.value.lower())
    
    @property
    def domain(self) -> str:
        """Extract domain from email address."""
        return self.value.split('@')[1]
    
    @property
    def local_part(self) -> str:
        """Extract local part (before @) from email address."""
        return self.value.split('@')[0]
    
    def __str__(self) -> str:
        """String representation of email."""
        return self.value
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Email('{self.value}')"
