"""
Celery tasks for async video encoding.

These tasks handle background video transcoding using either AWS MediaConvert
or local FFmpeg processing.
"""

import logging
from typing import List
from uuid import UUID

from celery import Task
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.models.video import Video
from app.videos.models.encoding_job import EncodingStatus
from app.videos.services.advanced_encoding_service import (
    AdvancedVideoEncodingService,
    EncodingPreset,
)

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session."""
    
    _db_session = None

    @property
    def db_session(self):
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = AsyncSessionLocal()
        return self._db_session

    def after_return(self, *args, **kwargs):
        """Cleanup database session after task completion."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None


@celery_app.task(
    name="videos.encode_video",
    bind=True,
    base=DatabaseTask,
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
)
async def encode_video_task(
    self,
    video_id: str,
    input_path: str,
    qualities: List[str] = None,
    output_format: str = "hls",
) -> dict:
    """
    Celery task to encode video in background.
    
    Args:
        video_id: UUID of the video to encode
        input_path: S3 key or local path to input video
        qualities: List of quality presets (240p, 480p, 720p, 1080p, 4k)
        output_format: Output format (hls or dash)
    
    Returns:
        Dictionary with encoding results
    """
    logger.info(f"Starting video encoding task for video {video_id}")
    
    # Convert qualities from strings to EncodingPreset enums
    if qualities is None:
        quality_presets = [
            EncodingPreset.MOBILE_240P,
            EncodingPreset.SD_480P,
            EncodingPreset.HD_720P,
            EncodingPreset.FULL_HD_1080P,
        ]
    else:
        quality_presets = [EncodingPreset(q) for q in qualities]

    try:
        async with AsyncSessionLocal() as db:
            # Initialize encoding service
            encoding_service = AdvancedVideoEncodingService(db)
            
            # Convert string UUID to UUID object
            video_uuid = UUID(video_id)
            
            # Start encoding
            result = await encoding_service.encode_video(
                video_id=video_uuid,
                input_path=input_path,
                qualities=quality_presets,
                output_format=output_format,
            )
            
            # Update video record with encoding results
            video_result = await db.execute(
                select(Video).where(Video.id == video_uuid)
            )
            video = video_result.scalar_one_or_none()
            
            if video:
                video.is_encoded = True
                video.hls_manifest_url = result.get("hls_manifest")
                video.dash_manifest_url = result.get("dash_manifest")
                await db.commit()
            
            logger.info(f"Video encoding completed for video {video_id}")
            return {
                "status": "success",
                "video_id": video_id,
                "result": result,
            }

    except Exception as e:
        logger.error(f"Video encoding failed for {video_id}: {e}", exc_info=True)
        
        # Retry the task
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for video {video_id}")
            
            # Mark as failed in database
            async with AsyncSessionLocal() as db:
                video_result = await db.execute(
                    select(Video).where(Video.id == UUID(video_id))
                )
                video = video_result.scalar_one_or_none()
                if video:
                    video.is_encoded = False
                    await db.commit()
            
            return {
                "status": "failed",
                "video_id": video_id,
                "error": str(e),
            }


@celery_app.task(
    name="videos.generate_thumbnails",
    bind=True,
    base=DatabaseTask,
    max_retries=2,
)
async def generate_thumbnails_task(
    self,
    video_id: str,
    input_path: str,
    count: int = 5,
) -> dict:
    """
    Generate video thumbnails in background.
    
    Args:
        video_id: UUID of the video
        input_path: S3 key or local path to video
        count: Number of thumbnails to generate
    
    Returns:
        Dictionary with thumbnail S3 keys
    """
    logger.info(f"Generating thumbnails for video {video_id}")
    
    try:
        async with AsyncSessionLocal() as db:
            encoding_service = AdvancedVideoEncodingService(db)
            video_uuid = UUID(video_id)
            
            # Generate thumbnails
            thumbnail_keys = await encoding_service.generate_thumbnails(
                video_id=video_uuid,
                input_path=input_path,
                count=count,
            )
            
            # Update video record
            video_result = await db.execute(
                select(Video).where(Video.id == video_uuid)
            )
            video = video_result.scalar_one_or_none()
            
            if video and thumbnail_keys:
                # Set the first thumbnail as the default
                video.thumbnail_url = thumbnail_keys[0]
                await db.commit()
            
            logger.info(f"Thumbnails generated for video {video_id}: {len(thumbnail_keys)} images")
            return {
                "status": "success",
                "video_id": video_id,
                "thumbnails": thumbnail_keys,
            }

    except Exception as e:
        logger.error(f"Thumbnail generation failed for {video_id}: {e}")
        
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            return {
                "status": "failed",
                "video_id": video_id,
                "error": str(e),
            }


@celery_app.task(name="videos.check_encoding_status")
async def check_encoding_status_task(job_id: str) -> dict:
    """
    Check encoding job status (for AWS MediaConvert jobs).
    
    Args:
        job_id: Encoding job ID
    
    Returns:
        Job status dictionary
    """
    async with AsyncSessionLocal() as db:
        encoding_service = AdvancedVideoEncodingService(db)
        
        try:
            progress = await encoding_service.get_encoding_progress(UUID(job_id))
            return progress
        except Exception as e:
            logger.error(f"Failed to check encoding status: {e}")
            return {
                "status": "error",
                "error": str(e),
            }


@celery_app.task(name="videos.cleanup_failed_encodings")
async def cleanup_failed_encodings_task(days: int = 7) -> dict:
    """
    Clean up failed encoding jobs older than specified days.
    
    Args:
        days: Age threshold in days
    
    Returns:
        Cleanup statistics
    """
    from datetime import datetime, timedelta, timezone
    from app.videos.models.encoding_job import EncodingJob
    
    logger.info(f"Cleaning up failed encodings older than {days} days")
    
    async with AsyncSessionLocal() as db:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Find old failed jobs
        result = await db.execute(
            select(EncodingJob).where(
                EncodingJob.status == EncodingStatus.FAILED,
                EncodingJob.created_at < cutoff_date,
            )
        )
        failed_jobs = result.scalars().all()
        
        # Delete them
        deleted_count = 0
        for job in failed_jobs:
            await db.delete(job)
            deleted_count += 1
        
        await db.commit()
        
        logger.info(f"Cleaned up {deleted_count} failed encoding jobs")
        return {
            "status": "success",
            "deleted_count": deleted_count,
        }
