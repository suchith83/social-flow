"""
Password Hasher - Auth Infrastructure

Handles password hashing and verification using bcrypt directly.
Provides secure password storage and validation.
"""

import bcrypt


class PasswordHasher:
    """
    Password hashing service using bcrypt.
    
    Provides secure password hashing with configurable rounds.
    """
    
    def __init__(self, rounds: int = 12):
        """
        Initialize password hasher.
        
        Args:
            rounds: Number of bcrypt rounds (default: 12)
                   Higher = more secure but slower
        """
        self._rounds = rounds
    
    def hash(self, password: str) -> str:
        """
        Hash a plain text password.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Hashed password as a string
            
        Raises:
            ValueError: If password is empty
        """
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Convert password to bytes (bcrypt has 72 byte limit)
        password_bytes = password.encode('utf-8')
        
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=self._rounds)
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        # Return as string
        return hashed.decode('utf-8')
    
    def verify(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Hashed password to check against
            
        Returns:
            True if password matches hash, False otherwise
        """
        if not password or not password_hash:
            return False
        
        try:
            password_bytes = password.encode('utf-8')
            hash_bytes = password_hash.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except (ValueError, AttributeError):
            return False
    
    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if a password hash needs to be rehashed.
        
        This is useful when you increase the number of rounds
        and want to upgrade existing hashes.
        
        Args:
            password_hash: The hash to check
            
        Returns:
            True if hash should be regenerated, False otherwise
        """
        try:
            hash_bytes = password_hash.encode('utf-8')
            # Extract rounds from existing hash
            # bcrypt hash format: $2b$rounds$salt+hash
            parts = hash_bytes.split(b'$')
            if len(parts) >= 3:
                current_rounds = int(parts[2])
                return current_rounds < self._rounds
        except (ValueError, IndexError, AttributeError):
            pass
        
        return True


# Singleton instance for application use
password_hasher = PasswordHasher(rounds=12)
