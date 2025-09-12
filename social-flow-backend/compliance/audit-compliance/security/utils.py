# utils.py
import hashlib
import hmac
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional

# Configure a separate file for important security logs
logging.basicConfig(
    filename="security_audit.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def secure_hash(data: str, salt: Optional[str] = None) -> str:
    """
    Generate a SHA-256 HMAC (when salt provided) or plain SHA-256 digest.
    Use HMAC for keyed hashing to protect against length-extension attacks.
    """
    if salt:
        return hmac.new(salt.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).hexdigest()
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def timing_safe_compare(a: str, b: str) -> bool:
    """
    Constant-time string comparison to avoid timing attacks.
    Works for ASCII/UTF-8 strings (compares bytes to be safe).
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

def log_event(message: str, level: str = "INFO", **meta: Any):
    """
    Centralized structured logging for security events.
    Writes to file and prints to stdout for interactive usage.
    Meta can include context like 'user', 'ip', 'ticket_id'.
    """
    payload = {"timestamp": datetime.utcnow().isoformat(), "message": message, "meta": meta}
    if level == "ERROR":
        logging.error(json.dumps(payload))
    elif level == "WARNING":
        logging.warning(json.dumps(payload))
    else:
        logging.info(json.dumps(payload))
    # Print concise message for shell visibility (avoid sensitive fields)
    print(f"[{level}] {message}")

def load_json_config(path: str) -> Dict[str, Any]:
    """Simple config loader for policy/config files (JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
