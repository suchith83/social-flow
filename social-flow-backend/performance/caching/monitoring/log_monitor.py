import re
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

class CacheLogMonitor:
    """
    Monitors logs for cache-related issues such as errors, slow queries,
    or cache evictions.
    """

    ERROR_PATTERN = re.compile(r"(ERROR|Exception|Fail|Timeout)", re.IGNORECASE)
    EVICTION_PATTERN = re.compile(r"(Evict|Eviction)", re.IGNORECASE)

    def __init__(self, log_source: Iterable[str]):
        self.log_source = log_source

    def scan_logs(self):
        """Scan provided log lines for issues."""
        issues = []
        for line in self.log_source:
            if self.ERROR_PATTERN.search(line):
                logger.error(f"Cache error detected: {line.strip()}")
                issues.append(("error", line.strip()))
            elif self.EVICTION_PATTERN.search(line):
                logger.warning(f"Cache eviction detected: {line.strip()}")
                issues.append(("eviction", line.strip()))
        return issues
