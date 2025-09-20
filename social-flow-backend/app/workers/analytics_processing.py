"""
Analytics processing workers.

This module contains Celery tasks for analytics processing operations.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from celery import current_task
from app.workers.celery_app import celery_app
from app.services.analytics_service import analytics_service
from app.core.exceptions import AnalyticsServiceError

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.workers.analytics_processing.process_event")
def process_event_task(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process analytics event."""
    try:
        logger.info(f"Processing analytics event: {event_data.get('event_type')}")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "processing", "progress": 50})
        
        # Process event
        result = await analytics_service._process_event(event_data)
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing", "progress": 80})
        
        # Store processed event
        await analytics_service._store_processed_event(event_data, result)
        
        logger.info(f"Analytics event processed: {event_data.get('event_type')}")
        
        return {
            "status": "completed",
            "event_type": event_data.get('event_type'),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Analytics event processing failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.analytics_processing.generate_daily_reports")
def generate_daily_reports_task(self) -> Dict[str, Any]:
    """Generate daily analytics reports."""
    try:
        logger.info("Generating daily analytics reports")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "collecting_data", "progress": 30})
        
        # Collect daily data
        daily_data = await analytics_service._collect_daily_data()
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "generating_reports", "progress": 60})
        
        # Generate reports
        reports = await analytics_service._generate_daily_reports(daily_data)
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_reports", "progress": 80})
        
        # Store reports
        await analytics_service._store_daily_reports(reports)
        
        logger.info("Daily analytics reports generated")
        
        return {
            "status": "completed",
            "reports_generated": len(reports),
            "reports": reports
        }
        
    except Exception as e:
        logger.error(f"Daily reports generation failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.analytics_processing.cleanup_expired_sessions")
def cleanup_expired_sessions_task(self) -> Dict[str, Any]:
    """Clean up expired user sessions."""
    try:
        logger.info("Cleaning up expired sessions")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "identifying_expired", "progress": 30})
        
        # Identify expired sessions
        expired_sessions = await analytics_service._identify_expired_sessions()
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "cleaning_up", "progress": 60})
        
        # Clean up sessions
        cleaned_count = await analytics_service._cleanup_sessions(expired_sessions)
        
        logger.info(f"Cleaned up {cleaned_count} expired sessions")
        
        return {
            "status": "completed",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.analytics_processing.aggregate_metrics")
def aggregate_metrics_task(self, time_period: str = "hourly") -> Dict[str, Any]:
    """Aggregate analytics metrics."""
    try:
        logger.info(f"Aggregating {time_period} metrics")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "collecting_metrics", "progress": 30})
        
        # Collect metrics
        metrics = await analytics_service._collect_metrics(time_period)
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "aggregating", "progress": 60})
        
        # Aggregate metrics
        aggregated = await analytics_service._aggregate_metrics(metrics, time_period)
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing", "progress": 80})
        
        # Store aggregated metrics
        await analytics_service._store_aggregated_metrics(aggregated, time_period)
        
        logger.info(f"{time_period} metrics aggregated")
        
        return {
            "status": "completed",
            "time_period": time_period,
            "metrics_count": len(aggregated)
        }
        
    except Exception as e:
        logger.error(f"Metrics aggregation failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.analytics_processing.detect_anomalies")
def detect_anomalies_task(self) -> Dict[str, Any]:
    """Detect anomalies in analytics data."""
    try:
        logger.info("Detecting anomalies in analytics data")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "analyzing_data", "progress": 50})
        
        # Detect anomalies
        anomalies = await analytics_service._detect_anomalies()
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "storing_alerts", "progress": 80})
        
        # Store anomaly alerts
        await analytics_service._store_anomaly_alerts(anomalies)
        
        logger.info(f"Detected {len(anomalies)} anomalies")
        
        return {
            "status": "completed",
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.analytics_processing.update_user_profiles")
def update_user_profiles_task(self) -> Dict[str, Any]:
    """Update user profiles based on analytics data."""
    try:
        logger.info("Updating user profiles")
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "analyzing_behavior", "progress": 30})
        
        # Analyze user behavior
        behavior_data = await analytics_service._analyze_user_behavior()
        
        # Update task progress
        self.update_state(state="PROGRESS", meta={"status": "updating_profiles", "progress": 60})
        
        # Update profiles
        updated_count = await analytics_service._update_user_profiles(behavior_data)
        
        logger.info(f"Updated {updated_count} user profiles")
        
        return {
            "status": "completed",
            "updated_count": updated_count
        }
        
    except Exception as e:
        logger.error(f"User profile update failed: {e}")
        raise


@celery_app.task(bind=True, name="app.workers.analytics_processing.process_batch_analytics")
def process_batch_analytics_task(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process batch analytics operations."""
    try:
        logger.info(f"Processing batch analytics for {len(batch_data)} items")
        
        processed_count = 0
        failed_count = 0
        
        for i, item in enumerate(batch_data):
            try:
                # Update task progress
                progress = int((i / len(batch_data)) * 100)
                self.update_state(state="PROGRESS", meta={"status": "processing", "progress": progress})
                
                # Process item
                await process_event_task.delay(item)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process batch item {i}: {e}")
                failed_count += 1
        
        logger.info(f"Batch analytics processing completed: {processed_count} processed, {failed_count} failed")
        
        return {
            "status": "completed",
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_items": len(batch_data)
        }
        
    except Exception as e:
        logger.error(f"Batch analytics processing failed: {e}")
        raise
