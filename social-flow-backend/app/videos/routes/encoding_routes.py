"""
Video Encoding API Endpoints.

Provides REST API endpoints for triggering video encoding, checking status,
and managing encoding jobs.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.video import Video
from app.videos.models.encoding_job import EncodingJob, EncodingStatus, EncodingQuality
from app.videos.services.advanced_encoding_service import AdvancedVideoEncodingService, EncodingPreset
from app.videos.tasks.encoding_tasks import encode_video_task, generate_thumbnails_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/videos", tags=["video-encoding"])


# ============================================================================
# Request/Response Models
# ============================================================================

class EncodeVideoRequest(BaseModel):
    """Request model for encoding a video."""
    
    video_id: UUID = Field(..., description="UUID of the video to encode")
    qualities: Optional[List[str]] = Field(
        default=["240p", "480p", "720p", "1080p"],
        description="List of quality levels to encode (240p, 360p, 480p, 720p, 1080p, 4k)"
    )
    output_format: str = Field(
        default="hls",
        description="Output format: 'hls' for HTTP Live Streaming or 'dash' for MPEG-DASH"
    )
    async_processing: bool = Field(
        default=True,
        description="If true, encode asynchronously using Celery. If false, encode synchronously."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "qualities": ["240p", "480p", "720p", "1080p"],
                "output_format": "hls",
                "async_processing": True,
            }
        }


class EncodingJobResponse(BaseModel):
    """Response model for encoding job."""
    
    job_id: UUID
    video_id: UUID
    status: str
    progress: int
    input_path: str
    output_format: str
    output_paths: Optional[dict]
    hls_manifest_url: Optional[str]
    dash_manifest_url: Optional[str]
    mediaconvert_job_id: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class GenerateThumbnailsRequest(BaseModel):
    """Request model for generating thumbnails."""
    
    video_id: UUID = Field(..., description="UUID of the video")
    count: int = Field(default=5, ge=1, le=10, description="Number of thumbnails to generate")


class ThumbnailsResponse(BaseModel):
    """Response model for thumbnail generation."""
    
    video_id: UUID
    thumbnails: List[str] = Field(..., description="List of S3 keys for generated thumbnails")
    count: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/encode", response_model=EncodingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def encode_video(
    request: EncodeVideoRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EncodingJobResponse:
    """
    Encode a video to multiple quality levels.
    
    This endpoint triggers video encoding with the following features:
    - Multi-quality encoding (240p to 4K)
    - HLS or DASH adaptive streaming output
    - AWS MediaConvert (cloud) or FFmpeg (local) processing
    - Asynchronous processing with Celery
    
    **Requires authentication.**
    
    **Rate limit:** 10 requests per minute
    """
    logger.info(f"Encoding request for video {request.video_id} by user {current_user.id}")
    
    # Verify video exists and user owns it
    result = await db.execute(
        select(Video).where(Video.id == request.video_id)
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {request.video_id}"
        )
    
    if video.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to encode this video"
        )
    
    if not video.s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video file not uploaded yet"
        )
    
    # Validate quality options
    valid_qualities = {"240p", "360p", "480p", "720p", "1080p", "4k"}
    if not all(q in valid_qualities for q in request.qualities):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid quality. Valid options: {', '.join(valid_qualities)}"
        )
    
    # Validate output format
    if request.output_format not in {"hls", "dash"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid output format. Must be 'hls' or 'dash'"
        )
    
    if request.async_processing:
        # Trigger async encoding with Celery
        task = encode_video_task.delay(
            video_id=str(request.video_id),
            input_path=video.s3_key,
            qualities=request.qualities,
            output_format=request.output_format,
        )
        
        # Create encoding job record
        encoding_job = EncodingJob(
            video_id=request.video_id,
            status=EncodingStatus.QUEUED,
            input_path=video.s3_key,
            output_format=request.output_format,
        )
        
        db.add(encoding_job)
        await db.commit()
        await db.refresh(encoding_job)
        
        logger.info(f"Async encoding task queued: {task.id} for video {request.video_id}")
        
        return EncodingJobResponse(
            job_id=encoding_job.id,
            video_id=encoding_job.video_id,
            status=encoding_job.status.value,
            progress=encoding_job.progress,
            input_path=encoding_job.input_path,
            output_format=encoding_job.output_format,
            output_paths=encoding_job.output_paths,
            hls_manifest_url=encoding_job.hls_manifest_url,
            dash_manifest_url=encoding_job.dash_manifest_url,
            mediaconvert_job_id=encoding_job.mediaconvert_job_id,
            started_at=encoding_job.started_at.isoformat() if encoding_job.started_at else None,
            completed_at=encoding_job.completed_at.isoformat() if encoding_job.completed_at else None,
            error_message=encoding_job.error_message,
        )
    
    else:
        # Synchronous encoding (blocks until complete)
        encoding_service = AdvancedVideoEncodingService(db)
        
        quality_presets = [EncodingPreset(q) for q in request.qualities]
        
        try:
            result = await encoding_service.encode_video(
                video_id=request.video_id,
                input_path=video.s3_key,
                qualities=quality_presets,
                output_format=request.output_format,
            )
            
            # Get the created encoding job
            job_result = await db.execute(
                select(EncodingJob)
                .where(EncodingJob.video_id == request.video_id)
                .order_by(EncodingJob.created_at.desc())
                .limit(1)
            )
            encoding_job = job_result.scalar_one()
            
            return EncodingJobResponse(
                job_id=encoding_job.id,
                video_id=encoding_job.video_id,
                status=encoding_job.status.value,
                progress=100,
                input_path=encoding_job.input_path,
                output_format=encoding_job.output_format,
                output_paths=encoding_job.output_paths,
                hls_manifest_url=encoding_job.hls_manifest_url,
                dash_manifest_url=encoding_job.dash_manifest_url,
                mediaconvert_job_id=encoding_job.mediaconvert_job_id,
                started_at=encoding_job.started_at.isoformat() if encoding_job.started_at else None,
                completed_at=encoding_job.completed_at.isoformat() if encoding_job.completed_at else None,
                error_message=encoding_job.error_message,
            )
            
        except Exception as e:
            logger.error(f"Synchronous encoding failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Encoding failed: {str(e)}"
            )


@router.get("/encoding-jobs/{job_id}", response_model=EncodingJobResponse)
async def get_encoding_job_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EncodingJobResponse:
    """
    Get the status of an encoding job.
    
    Returns detailed information about an encoding job including:
    - Current status (pending, processing, completed, failed)
    - Progress percentage
    - Output paths and manifest URLs
    - Error messages (if failed)
    
    **Requires authentication.**
    """
    result = await db.execute(
        select(EncodingJob).where(EncodingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Encoding job not found: {job_id}"
        )
    
    # Verify user owns the video
    video_result = await db.execute(
        select(Video).where(Video.id == job.video_id)
    )
    video = video_result.scalar_one_or_none()
    
    if not video or video.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this encoding job"
        )
    
    return EncodingJobResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        progress=job.progress,
        input_path=job.input_path,
        output_format=job.output_format,
        output_paths=job.output_paths,
        hls_manifest_url=job.hls_manifest_url,
        dash_manifest_url=job.dash_manifest_url,
        mediaconvert_job_id=job.mediaconvert_job_id,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
    )


@router.get("/videos/{video_id}/encoding-jobs", response_model=List[EncodingJobResponse])
async def get_video_encoding_jobs(
    video_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[EncodingJobResponse]:
    """
    Get all encoding jobs for a specific video.
    
    Returns a list of all encoding attempts for the video, sorted by creation date
    (newest first).
    
    **Requires authentication.**
    """
    # Verify video exists and user owns it
    video_result = await db.execute(
        select(Video).where(Video.id == video_id)
    )
    video = video_result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {video_id}"
        )
    
    if video.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view encoding jobs for this video"
        )
    
    # Get all encoding jobs
    result = await db.execute(
        select(EncodingJob)
        .where(EncodingJob.video_id == video_id)
        .order_by(EncodingJob.created_at.desc())
    )
    jobs = result.scalars().all()
    
    return [
        EncodingJobResponse(
            job_id=job.id,
            video_id=job.video_id,
            status=job.status.value,
            progress=job.progress,
            input_path=job.input_path,
            output_format=job.output_format,
            output_paths=job.output_paths,
            hls_manifest_url=job.hls_manifest_url,
            dash_manifest_url=job.dash_manifest_url,
            mediaconvert_job_id=job.mediaconvert_job_id,
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error_message=job.error_message,
        )
        for job in jobs
    ]


@router.post("/thumbnails", response_model=ThumbnailsResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_video_thumbnails(
    request: GenerateThumbnailsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ThumbnailsResponse:
    """
    Generate thumbnails for a video.
    
    Extracts frames from the video at even intervals and generates thumbnail images.
    The first thumbnail is automatically set as the video's default thumbnail.
    
    **Requires authentication.**
    
    **Rate limit:** 20 requests per minute
    """
    logger.info(f"Thumbnail generation request for video {request.video_id} by user {current_user.id}")
    
    # Verify video exists and user owns it
    result = await db.execute(
        select(Video).where(Video.id == request.video_id)
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found: {request.video_id}"
        )
    
    if video.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to generate thumbnails for this video"
        )
    
    if not video.s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video file not uploaded yet"
        )
    
    # Trigger async thumbnail generation
    task = generate_thumbnails_task.delay(
        video_id=str(request.video_id),
        input_path=video.s3_key,
        count=request.count,
    )
    
    logger.info(f"Thumbnail generation task queued: {task.id} for video {request.video_id}")
    
    return ThumbnailsResponse(
        video_id=request.video_id,
        thumbnails=[],  # Will be populated when task completes
        count=request.count,
    )


@router.delete("/encoding-jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_encoding_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Cancel an encoding job.
    
    Attempts to cancel a running or queued encoding job. Already completed
    jobs cannot be cancelled.
    
    **Requires authentication.**
    """
    result = await db.execute(
        select(EncodingJob).where(EncodingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Encoding job not found: {job_id}"
        )
    
    # Verify user owns the video
    video_result = await db.execute(
        select(Video).where(Video.id == job.video_id)
    )
    video = video_result.scalar_one_or_none()
    
    if not video or video.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to cancel this encoding job"
        )
    
    if job.status in (EncodingStatus.COMPLETED, EncodingStatus.FAILED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status.value}"
        )
    
    # Update status to canceled
    job.status = EncodingStatus.CANCELED
    await db.commit()
    
    logger.info(f"Encoding job {job_id} canceled by user {current_user.id}")
