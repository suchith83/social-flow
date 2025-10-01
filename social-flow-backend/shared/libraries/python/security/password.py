# password.py
import bcrypt


class PasswordHasher:
    """
    Secure password hashing and verification.
    Uses bcrypt (adaptive hashing).
    """

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed.encode())
