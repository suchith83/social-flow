"""
Utility helpers used across tests.
"""

import random
import string
from typing import Any, Dict
import time


def random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S")

def ensure_2xx(resp) -> bool:
    return 200 <= resp.status_code < 300
