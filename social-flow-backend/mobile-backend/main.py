from fastapi import FastAPI
import os
import sys

# allow local imports when running in repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mobile_backend.core.logger import get_logger
from mobile_backend.core.config import settings
from mobile_backend.api.routes import register_routes  # type: ignore

logger = get_logger("mobile-backend")
app = FastAPI(title="Mobile Backend (edge)", version="0.1.0")

register_routes(app)


@app.on_event("startup")
def startup():
    logger.info("Starting mobile-backend")
    logger.debug("Config: %s", {"ENV": settings.ENV, "PUSH_PROVIDER": settings.PUSH_PROVIDER})


@app.on_event("shutdown")
def shutdown():
    logger.info("Shutting down mobile-backend")
