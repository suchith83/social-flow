"""
Retry helper for flaky operations using tenacity.
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException

def retry_on_exception(max_attempts: int = 5, max_wait: int = 8):
    return retry(stop=stop_after_attempt(max_attempts), wait=wait_exponential(multiplier=1, max=max_wait),
                 retry=retry_if_exception_type(Exception))
