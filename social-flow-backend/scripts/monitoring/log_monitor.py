# scripts/monitoring/log_monitor.py
import logging
import time
import re
import threading
from typing import Dict, Any, List, Callable

logger = logging.getLogger("monitoring.logmonitor")


class LogMonitor:
    """
    Lightweight file tailing monitor looking for regex patterns and alerting on matches / thresholds.
    Not a full replacement for centralized logging, but useful for critical local logs.
    """

    def __init__(self, config: Dict[str, Any], alert_cb: Callable[[str], None] = None):
        self.config = config
        self.paths = config.get("monitoring", {}).get("logs", {}).get("paths", [])
        self.patterns = config.get("monitoring", {}).get("logs", {}).get("patterns", [])
        self.compiled = [re.compile(p) for p in self.patterns]
        self.alert_cb = alert_cb
        self.interval = int(config.get("monitoring", {}).get("logs", {}).get("interval", 5))
        self._stop = threading.Event()
        self._thread = None

    def _tail_file(self, path: str):
        """Simple tail implementation that yields new lines (works across restarts)."""
        try:
            with open(path, "r", errors="ignore") as fh:
                # move to EOF
                fh.seek(0, 2)
                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    yield line.rstrip("\n")
        except FileNotFoundError:
            logger.debug("Log file not found (will retry): %s", path)
            return
        except Exception:
            logger.exception("Tail failed for %s", path)
            return

    def _handle_line(self, line: str, source: str):
        for rx in self.compiled:
            if rx.search(line):
                msg = f"Log pattern matched in {source}: {line}"
                logger.warning(msg)
                if self.alert_cb:
                    try:
                        self.alert_cb(msg)
                    except Exception:
                        logger.exception("Alert callback failed")

    def _run_loop(self):
        logger.info("Starting log monitor for %d paths", len(self.paths))
        # create an iterator per path
        tails = {}
        while not self._stop.is_set():
            for path in self.paths:
                if path not in tails:
                    tails[path] = self._tail_file(path)
                try:
                    it = tails[path]
                    # consume available lines
                    for _ in range(100):  # avoid busy-looping too long
                        try:
                            line = next(it)
                        except StopIteration:
                            break
                        except TypeError:
                            break
                        if line:
                            self._handle_line(line, path)
                except Exception:
                    logger.exception("Error monitoring %s", path)
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
