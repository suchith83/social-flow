import logging
from typing import Optional

def get_logger(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    l = logging.getLogger(name)
    if not l.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        h.setFormatter(fmt)
        l.addHandler(h)
    l.setLevel(level)
    return l
