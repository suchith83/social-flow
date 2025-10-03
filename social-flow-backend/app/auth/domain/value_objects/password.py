"""
Password Value Object

Represents and validates passwords in the domain.
"""

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Password:
    """
    Password value object with validation.
    
    Immutable value object representing a password.
    Enforces password strength requirements.
    
    Note: This represents the plain-text password for validation.
    Always hash before storage using PasswordHasher.
    """
    
    value: str
    
    # Minimum password length
    MIN_LENGTH: ClassVar[int] = 8
    MAX_LENGTH: ClassVar[int] = 128
    
    def __post_init__(self) -> None:
        """Validate password strength on creation."""
        if not self.value:
            raise ValueError("Password cannot be empty")
        
        if len(self.value) < self.MIN_LENGTH:
            raise ValueError(f"Password must be at least {self.MIN_LENGTH} characters")
        
        if len(self.value) > self.MAX_LENGTH:
            raise ValueError(f"Password cannot exceed {self.MAX_LENGTH} characters")
        
        # Check for at least one letter
        if not re.search(r'[a-zA-Z]', self.value):
            raise ValueError("Password must contain at least one letter")
        
        # Check for at least one number
        if not re.search(r'\d', self.value):
            raise ValueError("Password must contain at least one number")
    
    def __str__(self) -> str:
        """String representation (masked for security)."""
        return "*" * len(self.value)
    
    def __repr__(self) -> str:
        """Developer-friendly representation (masked)."""
        return f"Password('{'*' * len(self.value)}')"
    
    @property
    def strength(self) -> str:
        """
        Calculate password strength.
        
        Returns:
            'weak', 'medium', or 'strong'
        """
        score = 0
        
        # Length bonus
        if len(self.value) >= 12:
            score += 2
        elif len(self.value) >= 10:
            score += 1
        
        # Character variety
        if re.search(r'[a-z]', self.value) and re.search(r'[A-Z]', self.value):
            score += 1
        
        if re.search(r'\d', self.value):
            score += 1
        
        if re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', self.value):
            score += 1
        
        # Return strength
        if score >= 4:
            return 'strong'
        elif score >= 2:
            return 'medium'
        else:
            return 'weak'
