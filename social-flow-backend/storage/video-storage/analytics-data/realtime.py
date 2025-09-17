"""
Real-time analytics streaming service
"""

import threading
import time
from .collector import AnalyticsCollector
from .processor import AnalyticsProcessor
from .aggregator import AnalyticsAggregator
from .utils import logger


class RealTimeAnalyticsService:
    def __init__(self):
        self.processor = AnalyticsProcessor()
        self.aggregator = AnalyticsAggregator()

    def start(self):
        logger.info("Starting RealTimeAnalyticsService...")

        def handler(event):
            processed = self.processor.process(event)
            if processed:
                self.aggregator.update(processed)

        collector = AnalyticsCollector(on_event=handler)
        threading.Thread(target=collector.run, daemon=True).start()

        while True:
            time.sleep(10)
            logger.info(f"Current metrics: {self.aggregator.metrics}")


if __name__ == "__main__":
    RealTimeAnalyticsService().start()
