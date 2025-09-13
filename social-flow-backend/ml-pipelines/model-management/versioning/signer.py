# Optional signing/verification for artifacts
# signer.py
"""
Optional artifact signer/verifier using HMAC or RSA (pluggable).
- HMAC signer uses a secret key to generate an artifact signature.
- RSA signer uses a private key to sign and a public key to verify.
This implementation includes simple HMAC signing for convenience.
"""

import hmac
import hashlib
from typing import Optional


class HMACSigner:
    def __init__(self, secret: bytes):
        self.secret = secret

    def sign(self, data: bytes) -> str:
        return hmac.new(self.secret, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)
