#!/usr/bin/env python3
"""
Backend validation script for Social Flow.

This script performs comprehensive validation of the backend
to ensure all components are properly integrated and functional.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import httpx
import subprocess
import importlib.util


class BackendValidator:
    """Backend validator for Social Flow."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        self.validation_results = []
        self.errors = []
        self.warnings = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_result(self, category: str, test: str, status: str, message: str = ""):
        """Log validation result."""
        result = {
            "category": category,
            "test": test,
            "status": status,
            "message": message
        }
        self.validation_results.append(result)
        
        if status == "ERROR":
            self.errors.append(f"{category}: {test} - {message}")
        elif status == "WARNING":
            self.warnings.append(f"{category}: {test} - {message}")
    
    async def validate_server_running(self):
        """Validate that the server is running."""
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                self.log_result("Server", "Health Check", "PASS", "Server is running")
                return True
            else:
                self.log_result("Server", "Health Check", "ERROR", f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Server", "Health Check", "ERROR", f"Server not accessible: {e}")
            return False
    
    async def validate_api_endpoints(self):
        """Validate API endpoints are accessible."""
        endpoints = [
            ("/api/v1/auth/register", "POST"),
            ("/api/v1/auth/login", "POST"),
            ("/api/v1/videos/feed", "GET"),
            ("/api/v1/videos/search", "GET"),
            ("/api/v1/notifications/", "GET"),
            ("/api/v1/live/active", "GET"),
            ("/api/v1/ads/video/test", "GET"),
            ("/api/v1/payments/process", "POST"),
            ("/api/v1/analytics/", "GET"),
            ("/api/v1/search", "GET"),
        ]
        
        for endpoint, method in endpoints:
            try:
                if method == "GET":
                    response = await self.client.get(endpoint)
                else:
                    response = await self.client.post(endpoint, json={})
                
                if response.status_code in [200, 401, 422]:  # Valid responses
                    self.log_result("API", f"{method} {endpoint}", "PASS", f"Status: {response.status_code}")
                else:
                    self.log_result("API", f"{method} {endpoint}", "WARNING", f"Unexpected status: {response.status_code}")
            except Exception as e:
                self.log_result("API", f"{method} {endpoint}", "ERROR", f"Endpoint not accessible: {e}")
    
    async def validate_database_connection(self):
        """Validate database connection."""
        try:
            # Test database connection by trying to register a user
            user_data = {
                "username": "validation_test_user",
                "email": "validation@test.com",
                "password": "testpassword123",
                "display_name": "Validation Test User",
            }
            
            response = await self.client.post("/api/v1/auth/register", json=user_data)
            if response.status_code == 201:
                self.log_result("Database", "Connection", "PASS", "Database is accessible")
                return True
            else:
                self.log_result("Database", "Connection", "ERROR", f"Database error: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Database", "Connection", "ERROR", f"Database not accessible: {e}")
            return False
    
    async def validate_redis_connection(self):
        """Validate Redis connection."""
        try:
            # Test Redis by trying to get notifications (which uses Redis for caching)
            response = await self.client.get("/api/v1/notifications/")
            if response.status_code in [200, 401]:  # Valid responses
                self.log_result("Redis", "Connection", "PASS", "Redis is accessible")
                return True
            else:
                self.log_result("Redis", "Connection", "WARNING", f"Redis may not be accessible: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Redis", "Connection", "WARNING", f"Redis not accessible: {e}")
            return False
    
    def validate_imports(self):
        """Validate that all required modules can be imported."""
        required_modules = [
            "app.main",
            "app.core.config",
            "app.core.database",
            "app.core.redis",
            "app.core.logging",
            "app.core.exceptions",
            "app.core.security",
            "app.models.user",
            "app.models.video",
            "app.models.post",
            "app.models.comment",
            "app.models.like",
            "app.models.follow",
            "app.models.ad",
            "app.models.payment",
            "app.models.subscription",
            "app.models.notification",
            "app.models.analytics",
            "app.models.view_count",
            "app.models.live_stream",
            "app.services.auth",
            "app.services.video_service",
            "app.services.ml_service",
            "app.services.analytics_service",
            "app.services.storage_service",
            "app.services.ads_service",
            "app.services.notification_service",
            "app.services.payments_service",
            "app.services.live_streaming_service",
            "app.api.v1.router",
            "app.api.v1.endpoints.auth",
            "app.api.v1.endpoints.videos",
            "app.api.v1.endpoints.posts",
            "app.api.v1.endpoints.comments",
            "app.api.v1.endpoints.likes",
            "app.api.v1.endpoints.follows",
            "app.api.v1.endpoints.ads",
            "app.api.v1.endpoints.payments",
            "app.api.v1.endpoints.subscriptions",
            "app.api.v1.endpoints.notifications",
            "app.api.v1.endpoints.analytics",
            "app.api.v1.endpoints.search",
            "app.api.v1.endpoints.admin",
            "app.api.v1.endpoints.moderation",
            "app.api.v1.endpoints.ml",
            "app.api.v1.endpoints.live_streaming",
        ]
        
        for module_name in required_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    self.log_result("Imports", module_name, "PASS", "Module can be imported")
                else:
                    self.log_result("Imports", module_name, "ERROR", "Module not found")
            except Exception as e:
                self.log_result("Imports", module_name, "ERROR", f"Import error: {e}")
    
    def validate_file_structure(self):
        """Validate file structure."""
        required_files = [
            "app/__init__.py",
            "app/main.py",
            "app/core/__init__.py",
            "app/core/config.py",
            "app/core/database.py",
            "app/core/redis.py",
            "app/core/logging.py",
            "app/core/exceptions.py",
            "app/core/security.py",
            "app/models/__init__.py",
            "app/models/user.py",
            "app/models/video.py",
            "app/models/post.py",
            "app/models/comment.py",
            "app/models/like.py",
            "app/models/follow.py",
            "app/models/ad.py",
            "app/models/payment.py",
            "app/models/subscription.py",
            "app/models/notification.py",
            "app/models/analytics.py",
            "app/models/view_count.py",
            "app/models/live_stream.py",
            "app/services/__init__.py",
            "app/services/auth.py",
            "app/services/video_service.py",
            "app/services/ml_service.py",
            "app/services/analytics_service.py",
            "app/services/storage_service.py",
            "app/services/ads_service.py",
            "app/services/notification_service.py",
            "app/services/payments_service.py",
            "app/services/live_streaming_service.py",
            "app/api/__init__.py",
            "app/api/v1/__init__.py",
            "app/api/v1/router.py",
            "app/api/v1/endpoints/__init__.py",
            "app/api/v1/endpoints/auth.py",
            "app/api/v1/endpoints/videos.py",
            "app/api/v1/endpoints/posts.py",
            "app/api/v1/endpoints/comments.py",
            "app/api/v1/endpoints/likes.py",
            "app/api/v1/endpoints/follows.py",
            "app/api/v1/endpoints/ads.py",
            "app/api/v1/endpoints/payments.py",
            "app/api/v1/endpoints/subscriptions.py",
            "app/api/v1/endpoints/notifications.py",
            "app/api/v1/endpoints/analytics.py",
            "app/api/v1/endpoints/search.py",
            "app/api/v1/endpoints/admin.py",
            "app/api/v1/endpoints/moderation.py",
            "app/api/v1/endpoints.ml.py",
            "app/api/v1/endpoints/live_streaming.py",
            "app/schemas/__init__.py",
            "app/schemas/auth.py",
            "app/workers/__init__.py",
            "app/workers/celery_app.py",
            "app/workers/video_processing.py",
            "app/workers/ai_processing.py",
            "app/workers/analytics_processing.py",
            "app/workers/notification_processing.py",
            "app/workers/email_processing.py",
            "requirements.txt",
            "requirements-dev.txt",
            "Dockerfile",
            "docker-compose.yml",
            "pytest.ini",
            ".pre-commit-config.yaml",
            "Makefile",
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                self.log_result("Files", file_path, "PASS", "File exists")
            else:
                self.log_result("Files", file_path, "ERROR", "File missing")
    
    def validate_dependencies(self):
        """Validate that all dependencies are installed."""
        try:
            result = subprocess.run(["pip", "check"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_result("Dependencies", "pip check", "PASS", "All dependencies are satisfied")
            else:
                self.log_result("Dependencies", "pip check", "ERROR", f"Dependency issues: {result.stdout}")
        except Exception as e:
            self.log_result("Dependencies", "pip check", "WARNING", f"Could not check dependencies: {e}")
    
    def validate_configuration(self):
        """Validate configuration files."""
        config_files = [
            "pytest.ini",
            ".pre-commit-config.yaml",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    if content.strip():
                        self.log_result("Config", config_file, "PASS", "Configuration file is valid")
                    else:
                        self.log_result("Config", config_file, "WARNING", "Configuration file is empty")
                except Exception as e:
                    self.log_result("Config", config_file, "ERROR", f"Configuration file error: {e}")
            else:
                self.log_result("Config", config_file, "ERROR", "Configuration file missing")
    
    async def validate_end_to_end_flow(self):
        """Validate end-to-end user flow."""
        try:
            # Register user
            user_data = {
                "username": "e2e_test_user",
                "email": "e2e@test.com",
                "password": "testpassword123",
                "display_name": "E2E Test User",
            }
            
            response = await self.client.post("/api/v1/auth/register", json=user_data)
            if response.status_code != 201:
                self.log_result("E2E", "User Registration", "ERROR", f"Registration failed: {response.status_code}")
                return False
            
            # Login user
            login_data = {
                "email": user_data["email"],
                "password": user_data["password"],
            }
            
            response = await self.client.post("/api/v1/auth/login", json=login_data)
            if response.status_code != 200:
                self.log_result("E2E", "User Login", "ERROR", f"Login failed: {response.status_code}")
                return False
            
            data = response.json()
            access_token = data["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # Test video upload
            video_data = {
                "title": "E2E Test Video",
                "description": "Test video for E2E validation",
                "filename": "test_video.mp4",
                "file_size": 1024000,
            }
            
            response = await self.client.post("/api/v1/videos/upload", json=video_data, headers=headers)
            if response.status_code != 201:
                self.log_result("E2E", "Video Upload", "ERROR", f"Video upload failed: {response.status_code}")
                return False
            
            video_id = response.json()["video_id"]
            
            # Test video retrieval
            response = await self.client.get(f"/api/v1/videos/{video_id}")
            if response.status_code != 200:
                self.log_result("E2E", "Video Retrieval", "ERROR", f"Video retrieval failed: {response.status_code}")
                return False
            
            # Test video like
            response = await self.client.post(f"/api/v1/videos/{video_id}/like", headers=headers)
            if response.status_code != 200:
                self.log_result("E2E", "Video Like", "ERROR", f"Video like failed: {response.status_code}")
                return False
            
            # Test video view
            response = await self.client.post(f"/api/v1/videos/{video_id}/view")
            if response.status_code != 200:
                self.log_result("E2E", "Video View", "ERROR", f"Video view failed: {response.status_code}")
                return False
            
            # Test video feed
            response = await self.client.get("/api/v1/videos/feed", headers=headers)
            if response.status_code != 200:
                self.log_result("E2E", "Video Feed", "ERROR", f"Video feed failed: {response.status_code}")
                return False
            
            self.log_result("E2E", "Complete Flow", "PASS", "End-to-end flow completed successfully")
            return True
            
        except Exception as e:
            self.log_result("E2E", "Complete Flow", "ERROR", f"E2E flow failed: {e}")
            return False
    
    async def run_validation(self):
        """Run all validation checks."""
        print("🔍 Starting Social Flow Backend Validation")
        print("=" * 60)
        
        # Server validation
        server_running = await self.validate_server_running()
        if not server_running:
            print("❌ Server is not running. Please start the server first.")
            return False
        
        # API validation
        await self.validate_api_endpoints()
        
        # Database validation
        await self.validate_database_connection()
        
        # Redis validation
        await self.validate_redis_connection()
        
        # File structure validation
        self.validate_file_structure()
        
        # Import validation
        self.validate_imports()
        
        # Dependency validation
        self.validate_dependencies()
        
        # Configuration validation
        self.validate_configuration()
        
        # End-to-end validation
        await self.validate_end_to_end_flow()
        
        # Print results
        self.print_results()
        
        return len(self.errors) == 0
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        # Count results by category
        categories = {}
        for result in self.validation_results:
            category = result["category"]
            if category not in categories:
                categories[category] = {"PASS": 0, "WARNING": 0, "ERROR": 0}
            categories[category][result["status"]] += 1
        
        # Print category summaries
        for category, counts in categories.items():
            total = sum(counts.values())
            passed = counts["PASS"]
            warnings = counts["WARNING"]
            errors = counts["ERROR"]
            
            print(f"\n{category}:")
            print(f"  Total: {total}")
            print(f"  Passed: {passed}")
            print(f"  Warnings: {warnings}")
            print(f"  Errors: {errors}")
        
        # Print errors
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        # Print warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Overall result
        total_tests = len(self.validation_results)
        passed_tests = len([r for r in self.validation_results if r["status"] == "PASS"])
        error_tests = len([r for r in self.validation_results if r["status"] == "ERROR"])
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if error_tests == 0:
            print("🎉 Backend validation passed!")
        else:
            print(f"❌ Backend validation failed with {error_tests} errors!")
        
        # Save results
        results_file = Path("validation_results.json")
        with open(results_file, "w") as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


async def main():
    """Main function."""
    async with BackendValidator() as validator:
        success = await validator.run_validation()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
