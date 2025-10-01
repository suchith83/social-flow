"""
AI/ML processing workers.

This module contains Celery tasks for AI/ML processing operations.
"""

import logging
from typing import Dict, Any, List
from app.workers.celery_app import celery_app
from app.ml.services.ml_service import ml_service

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.workers.ai_processing.analyze_content")
def analyze_content_task(self, content_id: str, content_type: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze content using ML models."""
    try:
        logger.info(f"Starting content analysis for {content_type} {content_id}")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "analyzing", "progress": 50})
        
        # Analyze content
        import asyncio
        analysis_results = asyncio.run(ml_service.analyze_content(content_type, content_data))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_results", "progress": 80})
        
        # Store analysis results
        asyncio.run(ml_service._store_analysis_results(content_id, content_type, analysis_results))
        
        logger.info(f"Content analysis completed for {content_type} {content_id}")
        
        return {
            "status": "completed",
            "content_id": content_id,
            "content_type": content_type,
            "analysis_results": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Content analysis failed for {content_type} {content_id}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.moderate_content")
def moderate_content_task(self, content_id: str, content_type: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Moderate content for safety and compliance."""
    try:
        logger.info(f"Starting content moderation for {content_type} {content_id}")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "moderating", "progress": 50})
        
        # Moderate content
        moderation_results = asyncio.run(ml_service.moderate_content(content_type, content_data))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_results", "progress": 80})
        
        # Store moderation results
        asyncio.run(ml_service._store_moderation_results(content_id, content_type, moderation_results))
        
        # Take action based on results
        if not moderation_results['is_safe']:
            asyncio.run(ml_service._handle_unsafe_content(content_id, content_type, moderation_results))
        
        logger.info(f"Content moderation completed for {content_type} {content_id}")
        
        return {
            "status": "completed",
            "content_id": content_id,
            "content_type": content_type,
            "moderation_results": moderation_results
        }
        
    except Exception as e:
        logger.error(f"Content moderation failed for {content_type} {content_id}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.generate_recommendations")
def generate_recommendations_task(self, user_id: str, content_type: str = "mixed", limit: int = 10) -> Dict[str, Any]:
    """Generate content recommendations for a user."""
    try:
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "generating", "progress": 50})
        
        # Generate recommendations
        recommendations = asyncio.run(ml_service.generate_recommendations(user_id, content_type, limit))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_results", "progress": 80})
        
        # Store recommendations
        asyncio.run(ml_service._store_recommendations(user_id, content_type, recommendations))
        
        logger.info(f"Recommendations generated for user {user_id}")
        
        return {
            "status": "completed",
            "user_id": user_id,
            "content_type": content_type,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Recommendation generation failed for user {user_id}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.generate_content")
def generate_content_task(self, content_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate content using ML models."""
    try:
        logger.info(f"Generating {content_type} content")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "generating", "progress": 50})
        
        # Generate content
        generated_content = asyncio.run(ml_service.generate_content(content_type, input_data))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_results", "progress": 80})
        
        # Store generated content
        asyncio.run(ml_service._store_generated_content(content_type, generated_content))
        
        logger.info(f"Content generation completed for {content_type}")
        
        return {
            "status": "completed",
            "content_type": content_type,
            "generated_content": generated_content
        }
        
    except Exception as e:
        logger.error(f"Content generation failed for {content_type}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.predict_viral_potential")
def predict_viral_potential_task(self, content_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict viral potential of content."""
    try:
        logger.info(f"Predicting viral potential for content {content_id}")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "predicting", "progress": 50})
        
        # Predict viral potential
        viral_prediction = asyncio.run(ml_service.predict_viral_potential(content_data))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_results", "progress": 80})
        
        # Store prediction
        asyncio.run(ml_service._store_viral_prediction(content_id, viral_prediction))
        
        logger.info(f"Viral prediction completed for content {content_id}")
        
        return {
            "status": "completed",
            "content_id": content_id,
            "viral_prediction": viral_prediction
        }
        
    except Exception as e:
        logger.error(f"Viral prediction failed for content {content_id}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.update_trending_content")
def update_trending_content_task(self) -> Dict[str, Any]:
    """Update trending content based on ML analysis."""
    try:
        logger.info("Updating trending content")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "analyzing_trends", "progress": 30})
        
        # Analyze current trends
        trends = asyncio.run(ml_service._analyze_current_trends())
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "updating_rankings", "progress": 60})
        
        # Update content rankings
        updated_count = asyncio.run(ml_service._update_content_rankings(trends))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_results", "progress": 80})
        
        # Store trending content
        asyncio.run(ml_service._store_trending_content(trends))
        
        logger.info(f"Trending content updated, {updated_count} items processed")
        
        return {
            "status": "completed",
            "updated_count": updated_count,
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"Trending content update failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.retrain_models")
def retrain_models_task(self, model_type: str = "all") -> Dict[str, Any]:
    """Retrain ML models with new data."""
    try:
        logger.info(f"Retraining {model_type} models")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "preparing_data", "progress": 20})
        
        # Prepare training data
        training_data = asyncio.run(ml_service._prepare_training_data(model_type))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "training", "progress": 50})
        
        # Train models
        training_results = asyncio.run(ml_service._train_models(model_type, training_data))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "evaluating", "progress": 80})
        
        # Evaluate models
        evaluation_results = asyncio.run(ml_service._evaluate_models(model_type, training_results))
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "deploying", "progress": 90})
        
        # Deploy updated models
        asyncio.run(ml_service._deploy_models(model_type, training_results))
        
        logger.info(f"Model retraining completed for {model_type}")
        
        return {
            "status": "completed",
            "model_type": model_type,
            "training_results": training_results,
            "evaluation_results": evaluation_results
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed for {model_type}: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.ai_processing.process_batch_ml")
def process_batch_ml_task(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process batch ML operations."""
    try:
        logger.info(f"Processing batch ML operations for {len(batch_data)} items")
        
        processed_count = 0
        failed_count = 0
        
        for i, item in enumerate(batch_data):
            try:
                # Update task progress
                progress = int((i / len(batch_data)) * 100)
                self.update_state(state="PROGRESS", meta={"status": "processing", "progress": progress})
                
                # Process item based on type
                if item['type'] == 'analyze':
                    asyncio.run(analyze_content_task.delay(
                        item['content_id'],
                        item['content_type'],
                        item['content_data']
                    ))
                elif item['type'] == 'moderate':
                    asyncio.run(moderate_content_task.delay(
                        item['content_id'],
                        item['content_type'],
                        item['content_data']
                    ))
                elif item['type'] == 'recommend':
                    asyncio.run(generate_recommendations_task.delay(
                        item['user_id'],
                        item.get('content_type', 'mixed')),
                        item.get('limit', 10)
                    )
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process batch item {i}: {e}")
                failed_count += 1
        
        logger.info(f"Batch ML processing completed: {processed_count} processed, {failed_count} failed")
        
        return {
            "status": "completed",
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_items": len(batch_data)
        }
        
    except Exception as e:
        logger.error(f"Batch ML processing failed: {e}")
        raise
