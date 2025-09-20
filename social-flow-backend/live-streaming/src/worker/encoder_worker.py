import threading
import time
from typing import Any

# Simple encoder worker that watches streams and simulates encoding readiness.
def _encoder_loop(manager, logger, poll_interval=5):
    while True:
        try:
            # find streams in "live" state and simulate encoder activity
            streams = manager.list_streams()
            for s in streams:
                sid = s["id"]
                # if stream is live and started_at is recent, ensure status remains "live"
                if s.get("status") == "live":
                    # simulate metric/logging
                    logger.info("Encoder worker heartbeat for stream %s", sid)
            time.sleep(poll_interval)
        except Exception:
            logger.exception("Encoder worker encountered an error")
            time.sleep(poll_interval)


def start_encoder_worker(manager, logger, poll_interval=5):
    t = threading.Thread(target=_encoder_loop, args=(manager, logger, poll_interval), daemon=True)
    t.start()
    return t
