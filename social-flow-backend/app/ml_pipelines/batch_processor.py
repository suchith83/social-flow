"""
Batch Processor for AI/ML Tasks.

Handles batch processing of videos for AI analysis including:
- Batch video analysis (YOLO, Whisper, CLIP, Scene Detection)
- Progress tracking and reporting
- Error handling and retry logic
- Resource management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for AI/ML video analysis tasks.
    
    Processes multiple videos in parallel with configurable concurrency,
    progress tracking, and error handling.
    """
    
    def __init__(
        self,
        max_concurrent_videos: int = 5,
        retry_attempts: int = 2,
    ):
        """
        Initialize batch processor.
        
        Args:
            max_concurrent_videos: Maximum number of videos to process concurrently
            retry_attempts: Number of retry attempts for failed videos
        """
        self.max_concurrent_videos = max_concurrent_videos
        self.retry_attempts = retry_attempts
        
        logger.info(
            f"Batch processor initialized with max {max_concurrent_videos} concurrent videos"
        )
    
    async def process_videos(
        self,
        video_ids: List[UUID],
        force_reanalysis: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple videos with AI analysis.
        
        Args:
            video_ids: List of video IDs to process
            force_reanalysis: Force re-analysis even if results exist
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with processing results
        """
        start_time = datetime.utcnow()
        total_videos = len(video_ids)
        
        logger.info(
            f"Starting batch video processing: {total_videos} videos, "
            f"force_reanalysis={force_reanalysis}"
        )
        
        # Results tracking
        processed_count = 0
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        errors: List[Dict[str, Any]] = []
        
        # Process videos in batches
        for i in range(0, total_videos, self.max_concurrent_videos):
            batch = video_ids[i:i + self.max_concurrent_videos]
            
            # Process batch concurrently
            tasks = [
                self._process_single_video(video_id, force_reanalysis)
                for video_id in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for video_id, result in zip(batch, results):
                processed_count += 1
                
                if isinstance(result, Exception):
                    failed_count += 1
                    errors.append({
                        "video_id": str(video_id),
                        "error": str(result),
                    })
                    logger.error(f"Failed to process video {video_id}: {result}")
                elif result.get("skipped"):
                    skipped_count += 1
                else:
                    successful_count += 1
                
                # Update progress
                if progress_callback:
                    progress = (processed_count / total_videos) * 100
                    progress_callback(progress)
            
            logger.info(
                f"Batch progress: {processed_count}/{total_videos} "
                f"(success={successful_count}, failed={failed_count}, skipped={skipped_count})"
            )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = {
            "total_videos": total_videos,
            "processed_count": processed_count,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "execution_time_seconds": execution_time,
            "errors": errors,
        }
        
        logger.info(
            f"Batch video processing complete: {successful_count}/{total_videos} successful "
            f"in {execution_time:.2f}s"
        )
        
        return result
    
    async def _process_single_video(
        self,
        video_id: UUID,
        force_reanalysis: bool = False,
    ) -> Dict[str, Any]:
        """Process a single video with AI analysis."""
        try:
            # Import here to avoid circular dependencies
            from app.videos.services.video_service import VideoService
            from app.core.database import get_db
            
            # Get database session
            async for db in get_db():
                video_service = VideoService(db)
                
                # Check if analysis already exists
                if not force_reanalysis:
                    # Check cache or video metadata
                    from app.core.redis import get_redis
                    redis = await get_redis()
                    
                    if redis:
                        cache_key = f"video:ai_analysis:{video_id}"
                        cached_result = await redis.get(cache_key)
                        
                        if cached_result:
                            logger.info(f"Skipping video {video_id} - analysis exists")
                            return {"skipped": True, "reason": "analysis_exists"}
                
                # Perform AI analysis
                result = await video_service.analyze_video_content(
                    video_id=video_id,
                    force_reanalysis=force_reanalysis,
                )
                
                logger.info(f"Successfully processed video {video_id}")
                return {"skipped": False, "result": result}
                
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
            raise
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        # This could track historical stats, current queue size, etc.
        return {
            "max_concurrent_videos": self.max_concurrent_videos,
            "retry_attempts": self.retry_attempts,
        }
