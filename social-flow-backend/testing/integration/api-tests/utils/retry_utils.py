"""
Common retry helpers for tests (wrap flaky external operations).
We use tenacity for declarative retry policies.
"""

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from requests.exceptions import RequestException, ConnectionError, Timeout

# Example decorator for retrying test-level operations (like polling)
retry_on_network = retry(stop=stop_after_attempt(5), wait=wait_fixed(1),
                         retry=retry_if_exception_type((ConnectionError, Timeout, RequestException)))
