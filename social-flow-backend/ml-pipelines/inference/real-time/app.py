# FastAPI entrypoint
# ================================================================
# File: app.py
# Purpose: Real-time inference FastAPI service
# ================================================================

import uvicorn
from fastapi import FastAPI

from router import router
from monitoring import setup_metrics
from middleware import setup_middleware
from utils import load_config, setup_logger
from model_loader import ModelLoader

logger = setup_logger("RealTimeApp")

# Load config
config = load_config("configs/real_time_inference.yaml")

# Initialize FastAPI
app = FastAPI(
    title="Real-Time Inference API",
    description="Low-latency ML inference service",
    version="1.0.0"
)

# Setup middleware (logging, tracing, rate limiting)
setup_middleware(app, config)

# Add routes
app.include_router(router)

# Setup monitoring
setup_metrics(app)

# Load & warm model
model_loader = ModelLoader(config["model"])
model = model_loader.load_model()
model_loader.warmup(model)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config["server"].get("host", "0.0.0.0"),
        port=config["server"].get("port", 8000),
        reload=config["server"].get("reload", False)
    )
