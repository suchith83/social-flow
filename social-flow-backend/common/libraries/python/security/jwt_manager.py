# jwt_manager.py
import jwt
import datetime
from typing import Dict


class JWTManager:
    """
    JSON Web Token (JWT) utility for authentication.
    """

    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret
        self.algorithm = algorithm

    def generate_token(self, payload: Dict, expiry_minutes: int = 60) -> str:
        payload["exp"] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiry_minutes)
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> Dict:
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
