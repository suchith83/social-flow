# utils.py
import logging
import hashlib

logging.basicConfig(
    filename="industry_compliance.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def log_event(message: str, level="INFO"):
    if level == "ERROR":
        logging.error(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ALERT":
        logging.critical(message)
    else:
        logging.info(message)
    print(f"[{level}] {message}")

def mask_sensitive(data: str) -> str:
    """Mask all but last 4 chars of sensitive data."""
    if not data:
        return "****"
    return "*" * (len(data)-4) + data[-4:]

def pseudonymize(identifier: str) -> str:
    """Hash an identifier for GDPR pseudonymization."""
    return hashlib.sha256(identifier.encode()).hexdigest()[:10]

def secure_hash(data: str) -> str:
    """Generate SHA-256 hash."""
    return hashlib.sha256(data.encode()).hexdigest()
