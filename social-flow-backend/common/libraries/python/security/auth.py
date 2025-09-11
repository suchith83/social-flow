# auth.py
from .jwt_manager import JWTManager
from .password import PasswordHasher


class Authenticator:
    """
    Authentication helper that combines password and JWT.
    """

    def __init__(self, jwt_secret: str):
        self.jwt_manager = JWTManager(jwt_secret)
        self.password_hasher = PasswordHasher()

    def register_user(self, username: str, password: str) -> dict:
        hashed_pw = self.password_hasher.hash_password(password)
        return {"username": username, "password_hash": hashed_pw}

    def login_user(self, username: str, password: str, stored_hash: str) -> str:
        if not self.password_hasher.verify_password(password, stored_hash):
            raise ValueError("Invalid credentials")
        return self.jwt_manager.generate_token({"sub": username})

    def validate_session(self, token: str) -> dict:
        return self.jwt_manager.validate_token(token)
