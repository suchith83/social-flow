"""
Antivirus scanning stub.

In production connect ClamAV or third-party scanning. Here we expose
a simple synchronous API that returns 'clean' or raises on detection.

This is intentionally pluggable.
"""

import random
from .utils import logger


class VirusDetected(Exception):
    pass


def scan_file(path: str) -> bool:
    """
    Stubbed scan: in real deployments call clamdscan or a scanning service.
    Returns True when clean, raises VirusDetected when infected.
    """
    # simulation: very low probability of infection
    if random.random() < 0.0001:
        logger.error(f"Virus detected in {path}")
        raise VirusDetected("Virus signature matched")
    logger.info(f"Scanned {path}: clean")
    return True
