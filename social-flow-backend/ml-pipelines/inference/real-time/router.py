# API endpoints
# ================================================================
# File: router.py
# Purpose: API endpoints
# ================================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from inference_service import InferenceService
from model_loader import ModelLoader
from utils import load_config

config = load_config("configs/real_time_inference.yaml")
model = ModelLoader(config["model"]).load_model()
service = InferenceService(model, config)

router = APIRouter()


class InferenceRequest(BaseModel):
    inputs: list


@router.post("/infer")
async def infer(req: InferenceRequest):
    try:
        results = await service.infer(req.inputs)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "ok"}
