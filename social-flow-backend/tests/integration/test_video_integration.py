"""
Integration tests for video functionality.

This module contains integration tests for the video
endpoints and service integration.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestVideoIntegration:
    """Integration tests for video endpoints."""

    def test_upload_video_success(self, client: TestClient, auth_headers, video_data):
        """Test successful video upload."""
        response = client.post("/api/v1/videos/upload", json=video_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert "video_id" in data

    def test_upload_video_unauthorized(self, client: TestClient, video_data):
        """Test video upload without authentication."""
        response = client.post("/api/v1/videos/upload", json=video_data)
        
        assert response.status_code == 401

    def test_upload_video_invalid_data(self, client: TestClient, auth_headers):
        """Test video upload with invalid data."""
        invalid_data = {
            "title": "",  # Empty title
            "description": "Test video description",
        }
        
        response = client.post("/api/v1/videos/upload", json=invalid_data, headers=auth_headers)
        
        assert response.status_code == 422

    def test_get_video_success(self, client: TestClient, test_video):
        """Test successful video retrieval."""
        response = client.get(f"/api/v1/videos/{test_video.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_video.id)
        assert data["title"] == test_video.title

    def test_get_video_not_found(self, client: TestClient):
        """Test video retrieval when video not found."""
        response = client.get("/api/v1/videos/nonexistent_video_id")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_update_video_success(self, client: TestClient, auth_headers, test_video):
        """Test successful video update."""
        updates = {
            "title": "Updated Video Title",
            "description": "Updated description",
        }
        
        response = client.put(f"/api/v1/videos/{test_video.id}", json=updates, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_update_video_unauthorized(self, client: TestClient, test_video):
        """Test video update without authentication."""
        updates = {
            "title": "Updated Video Title",
        }
        
        response = client.put(f"/api/v1/videos/{test_video.id}", json=updates)
        
        assert response.status_code == 401

    def test_delete_video_success(self, client: TestClient, auth_headers, test_video):
        """Test successful video deletion."""
        response = client.delete(f"/api/v1/videos/{test_video.id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_delete_video_unauthorized(self, client: TestClient, test_video):
        """Test video deletion without authentication."""
        response = client.delete(f"/api/v1/videos/{test_video.id}")
        
        assert response.status_code == 401

    def test_like_video_success(self, client: TestClient, auth_headers, test_video):
        """Test successful video like."""
        response = client.post(f"/api/v1/videos/{test_video.id}/like", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "liked"

    def test_like_video_unauthorized(self, client: TestClient, test_video):
        """Test video like without authentication."""
        response = client.post(f"/api/v1/videos/{test_video.id}/like")
        
        assert response.status_code == 401

    def test_unlike_video_success(self, client: TestClient, auth_headers, test_video):
        """Test successful video unlike."""
        response = client.delete(f"/api/v1/videos/{test_video.id}/like", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unliked"

    def test_view_video_success(self, client: TestClient, test_video):
        """Test successful video view."""
        response = client.post(f"/api/v1/videos/{test_video.id}/view")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "viewed"

    def test_get_video_feed_success(self, client: TestClient, auth_headers):
        """Test successful video feed retrieval."""
        response = client.get("/api/v1/videos/feed", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data

    def test_get_video_feed_unauthorized(self, client: TestClient):
        """Test video feed retrieval without authentication."""
        response = client.get("/api/v1/videos/feed")
        
        assert response.status_code == 401

    def test_search_videos_success(self, client: TestClient):
        """Test successful video search."""
        response = client.get("/api/v1/videos/search?q=test")
        
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data

    def test_search_videos_with_filters(self, client: TestClient):
        """Test video search with filters."""
        response = client.get("/api/v1/videos/search?q=test&category=gaming&duration_min=60")
        
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data

    def test_get_video_analytics_success(self, client: TestClient, auth_headers, test_video):
        """Test successful video analytics retrieval."""
        response = client.get(f"/api/v1/videos/{test_video.id}/analytics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "video_id" in data
        assert "views" in data

    def test_get_video_analytics_unauthorized(self, client: TestClient, test_video):
        """Test video analytics retrieval without authentication."""
        response = client.get(f"/api/v1/videos/{test_video.id}/analytics")
        
        assert response.status_code == 401

    def test_transcode_video_success(self, client: TestClient, auth_headers, test_video):
        """Test successful video transcoding."""
        transcode_data = {
            "quality": "720p",
        }
        
        response = client.post(f"/api/v1/videos/{test_video.id}/transcode", json=transcode_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "transcoding"

    def test_generate_thumbnails_success(self, client: TestClient, auth_headers, test_video):
        """Test successful thumbnail generation."""
        response = client.post(f"/api/v1/videos/{test_video.id}/thumbnails", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "thumbnails" in data

    def test_create_streaming_manifest_success(self, client: TestClient, auth_headers, test_video):
        """Test successful streaming manifest creation."""
        response = client.post(f"/api/v1/videos/{test_video.id}/manifest", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "manifest_url" in data

    def test_optimize_for_mobile_success(self, client: TestClient, auth_headers, test_video):
        """Test successful mobile optimization."""
        response = client.post(f"/api/v1/videos/{test_video.id}/optimize-mobile", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "mobile_url" in data

    def test_upload_chunk_success(self, client: TestClient, auth_headers):
        """Test successful chunk upload."""
        upload_data = {
            "filename": "test_video.mp4",
            "file_size": 1024000,
            "chunk_size": 1024,
        }
        
        response = client.post("/api/v1/videos/upload/initiate", json=upload_data, headers=auth_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert "upload_id" in data
        assert "upload_urls" in data

    def test_upload_chunk_unauthorized(self, client: TestClient):
        """Test chunk upload without authentication."""
        upload_data = {
            "filename": "test_video.mp4",
            "file_size": 1024000,
        }
        
        response = client.post("/api/v1/videos/upload/initiate", json=upload_data)
        
        assert response.status_code == 401

    def test_complete_upload_success(self, client: TestClient, auth_headers):
        """Test successful upload completion."""
        complete_data = {
            "upload_id": "test_upload_123",
            "chunks": [{"chunk_number": 1, "etag": "test_etag"}],
        }
        
        response = client.post("/api/v1/videos/upload/test_upload_123/complete", json=complete_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_upload_progress_success(self, client: TestClient, auth_headers):
        """Test successful upload progress retrieval."""
        response = client.get("/api/v1/videos/upload/test_upload_123/progress", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert "progress" in data
