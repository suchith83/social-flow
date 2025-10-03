"""
Username Value Object

Represents and validates usernames in the domain.
"""

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Username:
    """
    Username value object with validation.
    
    Immutable value object representing a username.
    Enforces username format rules and constraints.
    """
    
    value: str
    
    # Username validation regex pattern
    # Alphanumeric + underscore, must start with letter
    USERNAME_REGEX: ClassVar[re.Pattern] = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]{2,49}$')
    
    # Reserved usernames that cannot be used
    RESERVED_USERNAMES: ClassVar[set] = {
        'admin', 'root', 'system', 'api', 'www', 'support',
        'help', 'info', 'null', 'undefined', 'user', 'users',
        'auth', 'login', 'register', 'signup', 'signin', 'signout',
        'logout', 'account', 'profile', 'settings', 'dashboard',
        'moderator', 'mod', 'administrator',
    }
    
    def __post_init__(self) -> None:
        """Validate username on creation."""
        if not self.value:
            raise ValueError("Username cannot be empty")
        
        if len(self.value) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if len(self.value) > 50:
            raise ValueError("Username cannot exceed 50 characters")
        
        if not self.USERNAME_REGEX.match(self.value):
            raise ValueError(
                f"Invalid username format: {self.value}. "
                "Username must start with a letter and contain only letters, "
                "numbers, and underscores"
            )
        
        if self.value.lower() in self.RESERVED_USERNAMES:
            raise ValueError(f"Username '{self.value}' is reserved")
    
    def __str__(self) -> str:
        """String representation of username."""
        return self.value
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Username('{self.value}')"
