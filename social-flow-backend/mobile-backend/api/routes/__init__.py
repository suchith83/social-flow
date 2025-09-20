from fastapi import FastAPI
from .mobile import router as mobile_router

def register_routes(app: FastAPI):
    app.include_router(mobile_router, prefix="/mobile", tags=["mobile"])
