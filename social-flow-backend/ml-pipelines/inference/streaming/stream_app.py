# Main entrypoint for streaming inference
# ================================================================
# File: stream_app.py
# Purpose: Entry point for streaming inference service
# ================================================================

import asyncio
import logging

from consumer import StreamConsumer
from producer import StreamProducer
from model_loader import ModelLoader
from inference_worker import InferenceWorker
from monitoring import Monitoring
from utils import load_config, setup_logger

logger = setup_logger("StreamApp")

config = load_config("configs/streaming_inference.yaml")


async def main():
    logger.info("🚀 Starting Streaming Inference Service")

    # Monitoring
    monitoring = Monitoring(config["monitoring"])

    # Load model
    model_loader = ModelLoader(config["model"])
    model = model_loader.load_model()
    model_loader.warmup(model)

    # Worker
    worker = InferenceWorker(model, config["inference"])

    # Producer
    producer = StreamProducer(config["producer"])

    # Consumer
    consumer = StreamConsumer(config["consumer"], worker, producer, monitoring)

    # Run consumer loop
    await consumer.start()


if __name__ == "__main__":
    asyncio.run(main())
