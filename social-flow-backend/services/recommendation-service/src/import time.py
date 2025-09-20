import time
import hmac
import hashlib
import base64
import json
from typing import Dict, Any, Optional
from app.core.config import settings

PBKDF_ITER = 100_000
SALT_LEN = 16

def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

def _b64url_decode(s: str) -> bytes:
    s_bytes = s.encode("ascii")
    padding = b"=" * ((4 - len(s_bytes) % 4) % 4)
    return base64.urlsafe_b64decode(s_bytes + padding)

def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    if salt is None:
        salt = hashlib.sha256(str(time.time()).encode() + os_random(8)).digest()[:SALT_LEN]
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF_ITER)
    return {"salt": _b64url_encode(salt), "hash": _b64url_encode(dk)}

def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    salt = _b64url_decode(salt_b64)
    expected = _b64url_decode(hash_b64)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF_ITER)
    return hmac.compare_digest(dk, expected)

def os_random(n: int) -> bytes:
    # uses /dev/urandom or Windows CryptGenRandom via os.urandom
    import os
    return os.urandom(n)

def create_access_token(subject: Dict[str, Any], expires_in: Optional[int] = None) -> str:
    expires = int(time.time()) + (expires_in or settings.ACCESS_TOKEN_EXPIRE_SECONDS)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": subject, "exp": expires}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(settings.SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"

def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode()
        sig = _b64url_decode(sig_b64)
        expected = hmac.new(settings.SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid signature")
        payload = json.loads(_b64url_decode(payload_b64).decode())
        if int(time.time()) > int(payload.get("exp", 0)):
            raise ValueError("Token expired")
        return payload.get("sub", {})
    except Exception as exc:
        raise ValueError(f"Invalid token: {exc}")
