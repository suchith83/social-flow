"""
ML Tasks - Celery tasks for async ML processing.

These tasks run on the ai_processing queue (4 workers) for CPU-intensive ML operations.
"""

import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from celery import shared_task
from sqlalchemy import select, update

from app.core.database import async_session_maker
from app.videos.models.video import Video
from app.models.post import Post
from app.models.comment import Comment
from app.services.ml_service import ml_service


logger = logging.getLogger(__name__)


# ============================================================================
# CONTENT MODERATION TASKS
# ============================================================================

@shared_task(
    name="ml.moderate_video",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def moderate_video_task(self, video_id: str):
    """
    Moderate video content for safety and compliance.
    
    Checks for NSFW content, violence, and other policy violations.
    Updates video moderation status in database.
    """
    try:
        logger.info(f"Starting video moderation for video_id={video_id}")
        
        # Run async moderation
        import asyncio
        result = asyncio.run(_moderate_video_async(video_id))
        
        logger.info(f"Video moderation completed for video_id={video_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Video moderation failed for video_id={video_id}: {e}")
        self.retry(exc=e)


async def _moderate_video_async(video_id: str) -> Dict[str, Any]:
    """Async implementation of video moderation."""
    async with async_session_maker() as db:
        try:
            # Get video from database
            stmt = select(Video).where(Video.id == uuid.UUID(video_id))
            result = await db.execute(stmt)
            video = result.scalar_one_or_none()
            
            if not video:
                raise ValueError(f"Video not found: {video_id}")
            
            # Prepare content data for moderation
            content_data = {
                "video_url": video.url,
                "thumbnail_url": video.thumbnail_url,
                "title": video.title,
                "description": video.description,
            }
            
            # Run moderation
            moderation_result = await ml_service.moderate_content(
                content_type="video",
                content_data=content_data,
            )
            
            # Update video moderation status
            is_safe = moderation_result.get("is_safe", True)
            flags = moderation_result.get("flags", [])
            
            update_stmt = (
                update(Video)
                .where(Video.id == uuid.UUID(video_id))
                .values(
                    moderation_status="approved" if is_safe else "rejected",
                    moderation_flags=flags,
                    moderated_at=datetime.utcnow(),
                )
            )
            await db.execute(update_stmt)
            await db.commit()
            
            return {
                "video_id": video_id,
                "is_safe": is_safe,
                "flags": flags,
                "confidence": moderation_result.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            await db.rollback()
            raise


@shared_task(
    name="ml.moderate_post",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def moderate_post_task(self, post_id: str):
    """
    Moderate post content for spam, hate speech, etc.
    
    Checks text content and attached images for policy violations.
    """
    try:
        logger.info(f"Starting post moderation for post_id={post_id}")
        
        import asyncio
        result = asyncio.run(_moderate_post_async(post_id))
        
        logger.info(f"Post moderation completed for post_id={post_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Post moderation failed for post_id={post_id}: {e}")
        self.retry(exc=e)


async def _moderate_post_async(post_id: str) -> Dict[str, Any]:
    """Async implementation of post moderation."""
    async with async_session_maker() as db:
        try:
            # Get post from database
            stmt = select(Post).where(Post.id == uuid.UUID(post_id))
            result = await db.execute(stmt)
            post = result.scalar_one_or_none()
            
            if not post:
                raise ValueError(f"Post not found: {post_id}")
            
            # Prepare content data
            content_data = {
                "text": post.content,
                "image_url": post.image_url if hasattr(post, 'image_url') else None,
            }
            
            # Run moderation
            moderation_result = await ml_service.moderate_content(
                content_type="text",
                content_data=content_data,
            )
            
            # Update post moderation status
            is_safe = moderation_result.get("is_safe", True)
            flags = moderation_result.get("flags", [])
            
            update_stmt = (
                update(Post)
                .where(Post.id == uuid.UUID(post_id))
                .values(
                    is_flagged=not is_safe,
                    flag_reason=", ".join(flags) if flags else None,
                    moderated_at=datetime.utcnow(),
                )
            )
            await db.execute(update_stmt)
            await db.commit()
            
            return {
                "post_id": post_id,
                "is_safe": is_safe,
                "flags": flags,
                "confidence": moderation_result.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            await db.rollback()
            raise


@shared_task(
    name="ml.moderate_comment",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def moderate_comment_task(self, comment_id: str):
    """
    Moderate comment for spam, hate speech, etc.
    """
    try:
        logger.info(f"Starting comment moderation for comment_id={comment_id}")
        
        import asyncio
        result = asyncio.run(_moderate_comment_async(comment_id))
        
        logger.info(f"Comment moderation completed for comment_id={comment_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Comment moderation failed for comment_id={comment_id}: {e}")
        self.retry(exc=e)


async def _moderate_comment_async(comment_id: str) -> Dict[str, Any]:
    """Async implementation of comment moderation."""
    async with async_session_maker() as db:
        try:
            # Get comment from database
            stmt = select(Comment).where(Comment.id == uuid.UUID(comment_id))
            result = await db.execute(stmt)
            comment = result.scalar_one_or_none()
            
            if not comment:
                raise ValueError(f"Comment not found: {comment_id}")
            
            # Prepare content data
            content_data = {
                "text": comment.content,
            }
            
            # Run moderation
            moderation_result = await ml_service.moderate_content(
                content_type="text",
                content_data=content_data,
            )
            
            # Update comment moderation status
            is_safe = moderation_result.get("is_safe", True)
            flags = moderation_result.get("flags", [])
            
            update_stmt = (
                update(Comment)
                .where(Comment.id == uuid.UUID(comment_id))
                .values(
                    is_flagged=not is_safe,
                    flag_reason=", ".join(flags) if flags else None,
                    moderated_at=datetime.utcnow(),
                )
            )
            await db.execute(update_stmt)
            await db.commit()
            
            return {
                "comment_id": comment_id,
                "is_safe": is_safe,
                "flags": flags,
                "confidence": moderation_result.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            await db.rollback()
            raise


# ============================================================================
# CONTENT ANALYSIS TASKS
# ============================================================================

@shared_task(
    name="ml.analyze_content",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def analyze_content_task(self, content_id: str, content_type: str):
    """
    Analyze content to extract tags, categories, sentiment, etc.
    
    Args:
        content_id: ID of content (video, post, etc.)
        content_type: Type of content (video, post)
    """
    try:
        logger.info(f"Starting content analysis for {content_type}_id={content_id}")
        
        import asyncio
        result = asyncio.run(_analyze_content_async(content_id, content_type))
        
        logger.info(f"Content analysis completed for {content_type}_id={content_id}")
        return result
        
    except Exception as e:
        logger.error(f"Content analysis failed for {content_type}_id={content_id}: {e}")
        self.retry(exc=e)


async def _analyze_content_async(content_id: str, content_type: str) -> Dict[str, Any]:
    """Async implementation of content analysis."""
    async with async_session_maker() as db:
        try:
            # Get content from database
            if content_type == "video":
                stmt = select(Video).where(Video.id == uuid.UUID(content_id))
                result = await db.execute(stmt)
                content = result.scalar_one_or_none()
                
                if not content:
                    raise ValueError(f"Video not found: {content_id}")
                
                content_data = {
                    "text": f"{content.title} {content.description}",
                    "video_url": content.url,
                    "thumbnail_url": content.thumbnail_url,
                }
            
            elif content_type == "post":
                stmt = select(Post).where(Post.id == uuid.UUID(content_id))
                result = await db.execute(stmt)
                content = result.scalar_one_or_none()
                
                if not content:
                    raise ValueError(f"Post not found: {content_id}")
                
                content_data = {
                    "text": content.content,
                    "image_url": content.image_url if hasattr(content, 'image_url') else None,
                }
            
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Run analysis
            analysis_result = await ml_service.analyze_content(
                content_type=content_type,
                content_data=content_data,
            )
            
            # Store analysis results (could be in separate table or JSONB field)
            # For now, we'll just return the results
            
            return {
                f"{content_type}_id": content_id,
                "analysis": analysis_result,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            raise


# ============================================================================
# RECOMMENDATION TASKS
# ============================================================================

@shared_task(
    name="ml.update_recommendations",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def update_recommendations_task(self, user_id: Optional[str] = None):
    """
    Update recommendations for a user or all users.
    
    This is a periodic task that refreshes recommendation caches.
    """
    try:
        logger.info(f"Starting recommendation update for user_id={user_id or 'all'}")
        
        import asyncio
        result = asyncio.run(_update_recommendations_async(user_id))
        
        logger.info(f"Recommendation update completed for user_id={user_id or 'all'}")
        return result
        
    except Exception as e:
        logger.error(f"Recommendation update failed for user_id={user_id}: {e}")
        self.retry(exc=e)


async def _update_recommendations_async(user_id: Optional[str]) -> Dict[str, Any]:
    """Async implementation of recommendation update."""
    try:
        if user_id:
            # Update recommendations for specific user
            recommendations = await ml_service.generate_recommendations(
                user_id=user_id,
                content_type="mixed",
                limit=50,
            )
            
            # Cache recommendations in Redis
            # TODO: Store in Redis cache
            
            return {
                "user_id": user_id,
                "recommendations_count": len(recommendations),
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            # Update recommendations for all active users
            # TODO: Get list of active users and update recommendations
            return {
                "status": "batch_update_completed",
                "timestamp": datetime.utcnow().isoformat(),
            }
            
    except Exception:
        raise


@shared_task(
    name="ml.calculate_trending",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def calculate_trending_task(self, time_window: str = "24h"):
    """
    Calculate trending content scores.
    
    This is a periodic task that updates trending content rankings.
    """
    try:
        logger.info(f"Starting trending calculation for time_window={time_window}")
        
        import asyncio
        result = asyncio.run(_calculate_trending_async(time_window))
        
        logger.info("Trending calculation completed")
        return result
        
    except Exception as e:
        logger.error(f"Trending calculation failed: {e}")
        self.retry(exc=e)


async def _calculate_trending_async(time_window: str) -> Dict[str, Any]:
    """Async implementation of trending calculation."""
    try:
        # Get trending analysis
        trending_data = await ml_service.get_trending_analysis(time_window=time_window)
        
        # Store trending data in cache
        # TODO: Store in Redis cache
        
        return {
            "time_window": time_window,
            "rising_count": len(trending_data.get("rising", [])),
            "falling_count": len(trending_data.get("falling", [])),
            "stable_count": len(trending_data.get("stable", [])),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception:
        raise


# ============================================================================
# VIRAL PREDICTION TASKS
# ============================================================================

@shared_task(
    name="ml.predict_virality",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def predict_virality_task(self, content_id: str, content_type: str):
    """
    Predict viral potential of content.
    
    Analyzes content features and early engagement to predict virality.
    """
    try:
        logger.info(f"Starting viral prediction for {content_type}_id={content_id}")
        
        import asyncio
        result = asyncio.run(_predict_virality_async(content_id, content_type))
        
        logger.info(f"Viral prediction completed for {content_type}_id={content_id}")
        return result
        
    except Exception as e:
        logger.error(f"Viral prediction failed for {content_type}_id={content_id}: {e}")
        self.retry(exc=e)


async def _predict_virality_async(content_id: str, content_type: str) -> Dict[str, Any]:
    """Async implementation of viral prediction."""
    async with async_session_maker() as db:
        try:
            # Get content from database
            if content_type == "video":
                stmt = select(Video).where(Video.id == uuid.UUID(content_id))
                result = await db.execute(stmt)
                content = result.scalar_one_or_none()
                
                if not content:
                    raise ValueError(f"Video not found: {content_id}")
                
                content_data = {
                    "title": content.title,
                    "description": content.description,
                    "duration": content.duration,
                    "views": content.views_count,
                    "likes": content.likes_count,
                    "comments": content.comments_count,
                    "shares": content.shares_count,
                    "created_at": content.created_at.isoformat(),
                }
            
            elif content_type == "post":
                stmt = select(Post).where(Post.id == uuid.UUID(content_id))
                result = await db.execute(stmt)
                content = result.scalar_one_or_none()
                
                if not content:
                    raise ValueError(f"Post not found: {content_id}")
                
                content_data = {
                    "content": content.content,
                    "likes": content.likes_count,
                    "comments": content.comments_count,
                    "shares": content.shares_count,
                    "created_at": content.created_at.isoformat(),
                }
            
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Get viral prediction
            viral_prediction = await ml_service.predict_viral_potential(content_data)
            
            return {
                f"{content_type}_id": content_id,
                "viral_score": viral_prediction.get("viral_score", 0.0),
                "confidence": viral_prediction.get("confidence", 0.0),
                "factors": viral_prediction.get("factors", {}),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            raise


# ============================================================================
# CONTENT GENERATION TASKS
# ============================================================================

@shared_task(
    name="ml.generate_captions",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def generate_captions_task(self, video_id: str):
    """
    Generate automatic captions/subtitles for video.
    """
    try:
        logger.info(f"Starting caption generation for video_id={video_id}")
        
        import asyncio
        result = asyncio.run(_generate_captions_async(video_id))
        
        logger.info(f"Caption generation completed for video_id={video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Caption generation failed for video_id={video_id}: {e}")
        self.retry(exc=e)


async def _generate_captions_async(video_id: str) -> Dict[str, Any]:
    """Async implementation of caption generation."""
    async with async_session_maker() as db:
        try:
            # Get video from database
            stmt = select(Video).where(Video.id == uuid.UUID(video_id))
            result = await db.execute(stmt)
            video = result.scalar_one_or_none()
            
            if not video:
                raise ValueError(f"Video not found: {video_id}")
            
            # Generate captions
            caption_result = await ml_service.generate_content(
                content_type="caption",
                input_data={"video_url": video.url},
            )
            
            # Store captions
            # TODO: Store in database or S3
            
            return {
                "video_id": video_id,
                "caption": caption_result.get("caption"),
                "confidence": caption_result.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            raise


@shared_task(
    name="ml.generate_thumbnail",
    queue="ai_processing",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def generate_thumbnail_task(self, video_id: str):
    """
    Generate thumbnail for video using ML.
    
    Selects most engaging frame or generates custom thumbnail.
    """
    try:
        logger.info(f"Starting thumbnail generation for video_id={video_id}")
        
        import asyncio
        result = asyncio.run(_generate_thumbnail_async(video_id))
        
        logger.info(f"Thumbnail generation completed for video_id={video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Thumbnail generation failed for video_id={video_id}: {e}")
        self.retry(exc=e)


async def _generate_thumbnail_async(video_id: str) -> Dict[str, Any]:
    """Async implementation of thumbnail generation."""
    async with async_session_maker() as db:
        try:
            # Get video from database
            stmt = select(Video).where(Video.id == uuid.UUID(video_id))
            result = await db.execute(stmt)
            video = result.scalar_one_or_none()
            
            if not video:
                raise ValueError(f"Video not found: {video_id}")
            
            # Generate thumbnail
            thumbnail_result = await ml_service.generate_content(
                content_type="thumbnail",
                input_data={"video_url": video.url},
            )
            
            # Update video thumbnail URL
            thumbnail_url = thumbnail_result.get("thumbnail_url")
            if thumbnail_url:
                update_stmt = (
                    update(Video)
                    .where(Video.id == uuid.UUID(video_id))
                    .values(thumbnail_url=thumbnail_url)
                )
                await db.execute(update_stmt)
                await db.commit()
            
            return {
                "video_id": video_id,
                "thumbnail_url": thumbnail_url,
                "confidence": thumbnail_result.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception:
            await db.rollback()
            raise


# ============================================================================
# PERIODIC TASKS (configured in celerybeat schedule)
# ============================================================================

@shared_task(
    name="ml.batch_update_all_recommendations",
    queue="ai_processing",
)
def batch_update_all_recommendations_task():
    """
    Periodic task to update recommendations for all active users.
    
    Runs hourly via Celery Beat.
    """
    logger.info("Starting batch recommendation update for all users")
    return update_recommendations_task.delay(user_id=None)


@shared_task(
    name="ml.batch_calculate_trending",
    queue="ai_processing",
)
def batch_calculate_trending_task():
    """
    Periodic task to calculate trending content.
    
    Runs every 15 minutes via Celery Beat.
    """
    logger.info("Starting batch trending calculation")
    return calculate_trending_task.delay(time_window="24h")
