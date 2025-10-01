"""
Video processing tasks for Celery workers.

Handles video encoding, thumbnail generation, and related background tasks.
"""

import logging
from typing import Any, Dict

from celery import Task

from app.workers.celery_app import celery_app
from app.core.database import async_session_maker

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session management."""
    
    _session = None
    
    @property
    def session(self):
        if self._session is None:
            self._session = async_session_maker()
        return self._session


@celery_app.task(bind=True, name="app.workers.video_processing.process_video")
def process_video_task(self, video_id: str, user_id: str, s3_key: str) -> Dict[str, Any]:
    """Process uploaded video (transcoding, thumbnails, etc.)."""
    try:
        logger.info(f"Starting video processing for video {video_id}")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "downloading", "progress": 10})
        
        # Download video from S3
        video_content = storage_service.download_file(settings.AWS_S3_BUCKET, s3_key)
        
        self.update_state(state="PROGRESS", meta={"status": "extracting_metadata", "progress": 20})
        
        # Extract metadata
        metadata = video_service._extract_video_metadata_from_content(video_content)
        
        self.update_state(state="PROGRESS", meta={"status": "generating_thumbnails", "progress": 40})
        
        # Generate thumbnails
        thumbnails = video_service._generate_thumbnails_from_content(video_content, video_id)
        
        self.update_state(state="PROGRESS", meta={"status": "transcoding", "progress": 60})
        
        # Transcode video
        transcoded_urls = video_service._transcode_video_from_content(video_content, video_id)
        
        self.update_state(state="PROGRESS", meta={"status": "uploading_processed", "progress": 80})
        
        # Upload processed files to S3
        processed_urls = video_service._upload_processed_video(video_id, transcoded_urls, thumbnails)
        
        self.update_state(state="PROGRESS", meta={"status": "updating_database", "progress": 90})
        
        # Update database
        video_service._update_video_processing_complete(video_id, processed_urls, metadata)
        
        logger.info(f"Video processing completed for video {video_id}")
        
        return {
            "status": "completed",
            "video_id": video_id,
            "processed_urls": processed_urls,
            "thumbnails": thumbnails,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Video processing failed for video {video_id}: {e}")
        
        # Update database with error
        video_service._update_video_processing_failed(video_id, str(e))
        
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "video_id": video_id}
        )
        raise


@celery_app.task(bind=True, name="app.workers.video_processing.generate_thumbnails")
def generate_thumbnails_task(self, video_id: str, s3_key: str) -> Dict[str, Any]:
    """Generate thumbnails for a video."""
    try:
        logger.info(f"Generating thumbnails for video {video_id}")
        
        # Download video from S3
        video_content = storage_service.download_file(settings.AWS_S3_BUCKET, s3_key)
        
        # Generate thumbnails
        thumbnails = video_service._generate_thumbnails_from_content(video_content, video_id)
        
        # Upload thumbnails to S3
        thumbnail_urls = video_service._upload_thumbnails(video_id, thumbnails)
        
        # Update database
        video_service._update_video_thumbnails(video_id, thumbnail_urls)
        
        logger.info(f"Thumbnails generated for video {video_id}")
        
        return {
            "status": "completed",
            "video_id": video_id,
            "thumbnails": thumbnail_urls
        }
        
    except Exception as e:
        logger.error(f"Thumbnail generation failed for video {video_id}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.video_processing.transcode_video")
def transcode_video_task(self, video_id: str, s3_key: str, quality: str = "auto") -> Dict[str, Any]:
    """Transcode video to different qualities."""
    try:
        logger.info(f"Transcoding video {video_id} with quality {quality}")
        
        # Download video from S3
        video_content = storage_service.download_file(settings.AWS_S3_BUCKET, s3_key)
        
        # Transcode video
        transcoded_urls = video_service._transcode_video_from_content(video_content, video_id, quality)
        
        # Upload transcoded videos to S3
        processed_urls = video_service._upload_transcoded_video(video_id, transcoded_urls)
        
        # Update database
        video_service._update_video_transcoded(video_id, processed_urls)
        
        logger.info(f"Video transcoding completed for video {video_id}")
        
        return {
            "status": "completed",
            "video_id": video_id,
            "transcoded_urls": processed_urls
        }
        
    except Exception as e:
        logger.error(f"Video transcoding failed for video {video_id}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.video_processing.cleanup_failed_videos")
def cleanup_failed_videos_task(self) -> Dict[str, Any]:
    """Clean up failed video processing jobs."""
    try:
        logger.info("Starting cleanup of failed video processing jobs")
        
        # Get failed videos from database
        failed_videos = video_service._get_failed_videos()
        
        cleaned_count = 0
        for video in failed_videos:
            try:
                # Delete from S3
                storage_service.delete_file(settings.AWS_S3_BUCKET, video.s3_key)
                
                # Update database
                video_service._mark_video_deleted(video.id)
                
                cleaned_count += 1
                
            except Exception as e:
                logger.error(f"Failed to clean up video {video.id}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} failed videos")
        
        return {
            "status": "completed",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Failed video cleanup failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.video_processing.process_pending_videos")
def process_pending_videos_task(self) -> Dict[str, Any]:
    """Process all pending videos."""
    try:
        logger.info("Processing pending videos")
        
        # Get pending videos from database
        pending_videos = video_service._get_pending_videos()
        
        processed_count = 0
        for video in pending_videos:
            try:
                # Queue video processing task
                process_video_task.delay(str(video.id), str(video.owner_id), video.s3_key)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to queue video {video.id}: {e}")
        
        logger.info(f"Queued {processed_count} pending videos for processing")
        
        return {
            "status": "completed",
            "queued_count": processed_count
        }
        
    except Exception as e:
        logger.error(f"Pending video processing failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.video_processing.update_view_counts")
def update_view_counts_task(self) -> Dict[str, Any]:
    """Update view counts from cache to database."""
    try:
        logger.info("Updating view counts from cache to database")
        
        # Get view counts from cache
        view_counts = video_service._get_view_counts_from_cache()
        
        updated_count = 0
        for video_id, counts in view_counts.items():
            try:
                # Update database
                video_service._update_view_counts_in_database(video_id, counts)
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update view counts for video {video_id}: {e}")
        
        logger.info(f"Updated view counts for {updated_count} videos")
        
        return {
            "status": "completed",
            "updated_count": updated_count
        }
        
    except Exception as e:
        logger.error(f"View count update failed: {e}")
        raise
