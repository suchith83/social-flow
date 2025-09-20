from fastapi import FastAPI
from .health import router as health_router
from .users import router as users_router

def register_routes(app: FastAPI):
    app.include_router(health_router, prefix="", tags=["health"])
    app.include_router(users_router, prefix="/users", tags=["users"])
