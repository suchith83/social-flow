# crypto.py
import os
import hashlib
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hmac, hashes
from cryptography.hazmat.backends import default_backend


class Crypto:
    """
    Provides cryptographic utilities:
    - AES symmetric encryption (GCM mode)
    - HMAC signing
    - SHA hashing
    """

    def __init__(self, key: bytes = None):
        self.key = key or os.urandom(32)  # 256-bit AES key

    def encrypt(self, plaintext: str) -> dict:
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(self.key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext.encode()) + padder.finalize()

        ciphertext = encryptor.update(padded) + encryptor.finalize()
        return {
            "iv": base64.b64encode(iv).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "tag": base64.b64encode(encryptor.tag).decode()
        }

    def decrypt(self, data: dict) -> str:
        iv = base64.b64decode(data["iv"])
        ciphertext = base64.b64decode(data["ciphertext"])
        tag = base64.b64decode(data["tag"])

        cipher = Cipher(algorithms.AES(self.key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        return (unpadder.update(padded) + unpadder.finalize()).decode()

    @staticmethod
    def sha256(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def hmac_sign(self, data: str) -> str:
        h = hmac.HMAC(self.key, hashes.SHA256(), backend=default_backend())
        h.update(data.encode())
        return base64.b64encode(h.finalize()).decode()
