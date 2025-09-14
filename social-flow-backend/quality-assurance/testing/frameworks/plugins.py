"""
Helpers to register or create plugins for pytest.

Provides a programmatic way to add lightweight plugins such as:
- test selection by tag/marker via env var
- timing/per-test logging
- per-test retry (if plugin not installed)
"""

import logging
from typing import Dict

logger = logging.getLogger("qa-testing-frameworks.plugins")


def pytest_register_plugins(config: Dict = None):
    """
    Create small plugins and register them with pytest when using a programmatic
    pytest invocation. This function returns plugin objects that can be passed
    to pytest.main([...], plugins=[...]) if desired.

    For subprocess runner we cannot auto-register without a conftest.py: consider
    exposing plugin code as a module that tests import in conftest.
    """
    # Example plugin: simple test timer
    class TimerPlugin:
        def __init__(self):
            self.times = {}

        def pytest_runtest_protocol(self, item, nextitem):
            import time
            start = time.time()
            yield
            elapsed = time.time() - start
            self.times[item.nodeid] = elapsed
            logger.debug("Test %s took %.3fs", item.nodeid, elapsed)

    # More plugins can be constructed and returned here
    return {"timer": TimerPlugin()}
