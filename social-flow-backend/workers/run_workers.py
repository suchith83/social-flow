"""Start all background workers (video, ai, analytics) in one process for local development."""

import threading
import time
from workers.video_processing import start_worker as start_video_worker  # type: ignore
from workers.ai_processing import start_worker as start_ai_worker  # type: ignore
from workers.analytics_worker import start_worker as start_analytics_worker  # type: ignore
from workers.common import logger

def main():
    logger.info("Starting workers: video, ai, analytics")
    threads = []
    threads.append(start_video_worker())
    threads.append(start_ai_worker())
    threads.append(start_analytics_worker())

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down workers (main)")
        # threads are daemon; exit will stop them

if __name__ == "__main__":
    main()
