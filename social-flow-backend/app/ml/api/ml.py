"""
ML/AI endpoints.

This module contains all ML/AI related API endpoints.
"""

from typing import Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.auth.api.auth import get_current_active_user
from app.ml.services.ml_service import ml_service
from app.analytics.services.analytics_service import analytics_service
from app.ml.ai_processing import (
    analyze_content_task,
    moderate_content_task,
    generate_recommendations_task,
    generate_content_task,
    predict_viral_potential_task
)

router = APIRouter()


@router.post("/analyze")
async def analyze_content(
    content_type: str = Form(...),
    content_data: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Analyze content using ML models."""
    try:
        # Parse content data
        import json
        try:
            content_data_dict = json.loads(content_data)
        except json.JSONDecodeError:
            content_data_dict = {"text": content_data}
        
        # Queue analysis task
        task = analyze_content_task.delay(
            content_id=f"temp_{current_user.id}",
            content_type=content_type,
            content_data=content_data_dict
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_analysis_request",
            user_id=str(current_user.id),
            data={"content_type": content_type, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Content analysis queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Content analysis failed")


@router.post("/moderate")
async def moderate_content(
    content_type: str = Form(...),
    content_data: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Moderate content for safety and compliance."""
    try:
        # Parse content data
        import json
        try:
            content_data_dict = json.loads(content_data)
        except json.JSONDecodeError:
            content_data_dict = {"text": content_data}
        
        # Queue moderation task
        task = moderate_content_task.delay(
            content_id=f"temp_{current_user.id}",
            content_type=content_type,
            content_data=content_data_dict
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_moderation_request",
            user_id=str(current_user.id),
            data={"content_type": content_type, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Content moderation queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Content moderation failed")


@router.get("/recommendations")
async def get_recommendations(
    content_type: str = "mixed",
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get content recommendations for user."""
    try:
        # Queue recommendation task
        task = generate_recommendations_task.delay(
            user_id=str(current_user.id),
            content_type=content_type,
            limit=limit
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_recommendation_request",
            user_id=str(current_user.id),
            data={"content_type": content_type, "limit": limit, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Recommendations queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Recommendation generation failed")


@router.post("/generate")
async def generate_content(
    content_type: str = Form(...),
    input_data: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Generate content using ML models."""
    try:
        # Parse input data
        import json
        try:
            input_data_dict = json.loads(input_data)
        except json.JSONDecodeError:
            input_data_dict = {"text": input_data}
        
        # Queue generation task
        task = generate_content_task.delay(
            content_type=content_type,
            input_data=input_data_dict
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_generation_request",
            user_id=str(current_user.id),
            data={"content_type": content_type, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Content generation queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Content generation failed")


@router.post("/predict-viral")
async def predict_viral_potential(
    content_data: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Predict viral potential of content."""
    try:
        # Parse content data
        import json
        try:
            content_data_dict = json.loads(content_data)
        except json.JSONDecodeError:
            content_data_dict = {"text": content_data}
        
        # Queue prediction task
        task = predict_viral_potential_task.delay(
            content_id=f"temp_{current_user.id}",
            content_data=content_data_dict
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_viral_prediction_request",
            user_id=str(current_user.id),
            data={"task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Viral prediction queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Viral prediction failed")


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get status of ML task."""
    try:
        from app.workers.celery_app import celery_app
        
        # Get task result
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == "PENDING":
            return {
                "task_id": task_id,
                "status": "pending",
                "message": "Task is pending"
            }
        elif task_result.state == "PROGRESS":
            return {
                "task_id": task_id,
                "status": "processing",
                "progress": task_result.info.get("progress", 0),
                "message": task_result.info.get("status", "Processing...")
            }
        elif task_result.state == "SUCCESS":
            return {
                "task_id": task_id,
                "status": "completed",
                "result": task_result.result
            }
        else:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(task_result.info)
            }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get task status")


@router.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = Form("general"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Analyze uploaded image using ML models."""
    try:
        # Read image data
        image_data = await file.read()
        
        # Prepare content data
        content_data = {
            "image_data": image_data,
            "filename": file.filename,
            "content_type": file.content_type,
            "analysis_type": analysis_type
        }
        
        # Queue analysis task
        task = analyze_content_task.delay(
            content_id=f"image_{current_user.id}",
            content_type="image",
            content_data=content_data
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_image_analysis_request",
            user_id=str(current_user.id),
            data={"analysis_type": analysis_type, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Image analysis queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Image analysis failed")


@router.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    analysis_type: str = Form("general"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Analyze uploaded video using ML models."""
    try:
        # Read video data
        video_data = await file.read()
        
        # Prepare content data
        content_data = {
            "video_data": video_data,
            "filename": file.filename,
            "content_type": file.content_type,
            "analysis_type": analysis_type
        }
        
        # Queue analysis task
        task = analyze_content_task.delay(
            content_id=f"video_{current_user.id}",
            content_type="video",
            content_data=content_data
        )
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_video_analysis_request",
            user_id=str(current_user.id),
            data={"analysis_type": analysis_type, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": "Video analysis queued for processing"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Video analysis failed")


@router.get("/models/status")
async def get_models_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get status of ML models."""
    try:
        # Get model status from ML service
        model_status = await ml_service.get_models_status()
        
        return {
            "models": model_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get model status")


@router.post("/models/retrain")
async def retrain_models(
    model_type: str = "all",
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Retrain ML models."""
    try:
        from app.ml.ai_processing import retrain_models_task
        
        # Queue retraining task
        task = retrain_models_task.delay(model_type=model_type)
        
        # Track analytics event
        await analytics_service.track_event(
            event_type="ml_model_retrain_request",
            user_id=str(current_user.id),
            data={"model_type": model_type, "task_id": task.id}
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": f"Model retraining queued for {model_type}"
        }
        
    except Exception:
        raise HTTPException(status_code=500, detail="Model retraining failed")


# Enhanced recommendation endpoints from Python service

@router.get("/recommendations/{user_id}")
async def get_personalized_recommendations(
    user_id: str,
    limit: int = 50,
    algorithm: str = "hybrid",
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get personalized recommendations for a user."""
    try:
        result = await ml_service.get_personalized_recommendations(
            user_id=user_id, 
            limit=limit, 
            algorithm=algorithm
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@router.get("/similar-users/{user_id}")
async def get_similar_users(
    user_id: str,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get users similar to the given user."""
    try:
        result = await ml_service.get_similar_users(user_id=user_id, limit=limit)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get similar users")


@router.get("/similar-content/{content_id}")
async def get_similar_content(
    content_id: str,
    content_type: str = "video",
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get content similar to the given content."""
    try:
        result = await ml_service.get_similar_content(
            content_id=content_id, 
            content_type=content_type, 
            limit=limit
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get similar content")


@router.post("/feedback")
async def record_user_feedback(
    user_id: str,
    content_id: str,
    feedback_type: str,
    feedback_value: float,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Record user feedback for recommendation learning."""
    try:
        result = await ml_service.record_user_feedback(
            user_id=user_id,
            content_id=content_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.get("/viral-prediction/{content_id}")
async def get_viral_predictions(
    content_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Predict viral potential of content."""
    try:
        result = await ml_service.get_viral_predictions(content_id=content_id)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get viral predictions")


@router.get("/trending-analysis")
async def get_trending_analysis(
    time_window: str = "24h",
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get detailed trending analysis."""
    try:
        result = await ml_service.get_trending_analysis(time_window=time_window)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get trending analysis")

