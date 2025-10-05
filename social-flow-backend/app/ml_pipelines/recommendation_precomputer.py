"""
Recommendation Pre-computer.

Pre-computes and caches recommendations for users to improve response times.
Supports batch pre-computation and cache warming strategies.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from uuid import UUID

from app.core.redis import get_redis

logger = logging.getLogger(__name__)


class RecommendationPrecomputer:
    """
    Pre-computes recommendations for users in batch.
    
    Reduces latency by computing and caching recommendations ahead of time.
    Supports multiple recommendation algorithms and cache warming strategies.
    """
    
    def __init__(
        self,
        max_concurrent_users: int = 10,
        cache_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize recommendation pre-computer.
        
        Args:
            max_concurrent_users: Maximum number of users to process concurrently
            cache_ttl: Cache time-to-live in seconds
        """
        self.max_concurrent_users = max_concurrent_users
        self.cache_ttl = cache_ttl
        
        logger.info(
            f"Recommendation pre-computer initialized with max {max_concurrent_users} concurrent users"
        )
    
    async def precompute_recommendations(
        self,
        user_ids: List[UUID],
        algorithms: List[str] = ["hybrid"],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-compute recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs to pre-compute for
            algorithms: List of recommendation algorithms to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with pre-computation results
        """
        start_time = datetime.utcnow()
        total_users = len(user_ids)
        total_computations = total_users * len(algorithms)
        
        logger.info(
            f"Starting recommendation pre-computation: {total_users} users, "
            f"{len(algorithms)} algorithms = {total_computations} computations"
        )
        
        # Results tracking
        processed_count = 0
        successful_count = 0
        failed_count = 0
        cached_count = 0
        errors: List[Dict[str, Any]] = []
        
        # Process users in batches
        for i in range(0, total_users, self.max_concurrent_users):
            batch = user_ids[i:i + self.max_concurrent_users]
            
            # Process each user with all algorithms
            batch_tasks = []
            for user_id in batch:
                for algorithm in algorithms:
                    batch_tasks.append(
                        self._precompute_for_user(user_id, algorithm)
                    )
            
            # Execute batch concurrently
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Collect results
            for result in results:
                processed_count += 1
                
                if isinstance(result, Exception):
                    failed_count += 1
                    errors.append({"error": str(result)})
                elif result.get("cached"):
                    cached_count += 1
                    successful_count += 1
                else:
                    successful_count += 1
                
                # Update progress
                if progress_callback:
                    progress = (processed_count / total_computations) * 100
                    progress_callback(progress)
            
            logger.info(
                f"Batch progress: {processed_count}/{total_computations} "
                f"(success={successful_count}, failed={failed_count}, cached={cached_count})"
            )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = {
            "total_users": total_users,
            "total_algorithms": len(algorithms),
            "total_computations": total_computations,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "cached_count": cached_count,
            "execution_time_seconds": execution_time,
            "errors": errors,
        }
        
        logger.info(
            f"Recommendation pre-computation complete: {successful_count}/{total_computations} "
            f"successful in {execution_time:.2f}s"
        )
        
        return result
    
    async def _precompute_for_user(
        self,
        user_id: UUID,
        algorithm: str,
    ) -> Dict[str, Any]:
        """Pre-compute recommendations for a single user."""
        try:
            # Import here to avoid circular dependencies
            from app.services.recommendation_service import RecommendationService
            from app.core.database import get_db
            
            # Get database session
            async for db in get_db():
                rec_service = RecommendationService(db)
                
                # Get recommendations (will be cached automatically)
                recommendations = await rec_service.get_video_recommendations(
                    user_id=user_id,
                    limit=20,
                    algorithm=algorithm,
                    exclude_ids=None,
                )
                
                # Store in extended cache
                redis = await get_redis()
                if redis:
                    cache_key = f"recommendations:precomputed:{user_id}:{algorithm}"
                    import json
                    await redis.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(recommendations),
                    )
                
                logger.debug(
                    f"Pre-computed recommendations for user {user_id} "
                    f"with algorithm {algorithm}"
                )
                
                return {
                    "cached": True,
                    "user_id": str(user_id),
                    "algorithm": algorithm,
                    "count": len(recommendations.get("recommendations", [])),
                }
                
        except Exception as e:
            logger.error(
                f"Error pre-computing recommendations for user {user_id}: {e}",
                exc_info=True
            )
            raise
    
    async def warm_cache(
        self,
        limit: int = 1000,
        algorithm: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Warm recommendation cache for active users.
        
        Args:
            limit: Maximum number of users to warm cache for
            algorithm: Recommendation algorithm to use
            
        Returns:
            Dict with cache warming results
        """
        logger.info(f"Starting cache warming for {limit} users with {algorithm} algorithm")
        
        try:
            # Get active users from database
            from app.core.database import get_db
            from sqlalchemy import select, desc
            from app.users.models import User
            
            async for db in get_db():
                # Get most recently active users
                stmt = (
                    select(User.id)
                    .where(User.is_active.is_(True))
                    .order_by(desc(User.last_login))
                    .limit(limit)
                )
                
                result = await db.execute(stmt)
                user_ids = [row[0] for row in result.all()]
                
                # Pre-compute recommendations for these users
                precompute_result = await self.precompute_recommendations(
                    user_ids=user_ids,
                    algorithms=[algorithm],
                )
                
                logger.info(
                    f"Cache warming complete: {precompute_result['successful_count']} users"
                )
                
                return {
                    "precomputed_count": precompute_result['successful_count'],
                    "failed_count": precompute_result['failed_count'],
                    "execution_time": precompute_result['execution_time_seconds'],
                }
                
        except Exception as e:
            logger.error(f"Error warming cache: {e}", exc_info=True)
            return {
                "precomputed_count": 0,
                "failed_count": 0,
                "error": str(e),
            }
    
    async def invalidate_user_cache(
        self,
        user_id: UUID,
        algorithms: Optional[List[str]] = None,
    ) -> None:
        """
        Invalidate cached recommendations for a user.
        
        Args:
            user_id: User ID to invalidate cache for
            algorithms: Specific algorithms to invalidate (None = all)
        """
        try:
            redis = await get_redis()
            if not redis:
                return
            
            if algorithms is None:
                # Invalidate all algorithms
                pattern = f"recommendations:*:{user_id}:*"
                async for key in redis.scan_iter(match=pattern):
                    await redis.delete(key)
            else:
                # Invalidate specific algorithms
                for algorithm in algorithms:
                    keys = [
                        f"recommendations:feed:{user_id}:{algorithm}",
                        f"recommendations:videos:{user_id}:{algorithm}",
                        f"recommendations:precomputed:{user_id}:{algorithm}",
                    ]
                    for key in keys:
                        await redis.delete(key)
            
            logger.info(f"Invalidated recommendation cache for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}", exc_info=True)
