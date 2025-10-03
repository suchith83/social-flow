"""
Unit tests for video functionality.

This module contains unit tests for the video service
and related functionality.
"""

import pytest
from unittest.mock import Mock, patch
from app.videos.services.video_service import VideoService
from app.core.exceptions import VideoServiceError


class TestVideoService:
    """Test cases for VideoService."""

    @pytest.fixture
    def video_service(self):
        """Create VideoService instance for testing."""
        return VideoService()

    @pytest.mark.asyncio
    async def test_upload_video_success(self, video_service):
        """Test successful video upload."""
        video_data = {
            "title": "Test Video",
            "description": "Test video description",
            "filename": "test_video.mp4",
            "file_size": 1024000,
            "owner_id": "user123",
        }
        
        with patch.object(video_service, 'create_video') as mock_create:
            mock_create.return_value = {"id": "video123", "title": "Test Video"}
            
            result = await video_service.upload_video(video_data)
            
            assert result["id"] == "video123"
            assert result["title"] == "Test Video"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_video_invalid_format(self, video_service):
        """Test video upload with invalid format."""
        video_data = {
            "title": "Test Video",
            "filename": "test_video.txt",  # Invalid format
            "file_size": 1024000,
            "owner_id": "user123",
        }
        
        with pytest.raises(VideoServiceError):
            await video_service.upload_video(video_data)

    @pytest.mark.asyncio
    async def test_get_video_success(self, video_service):
        """Test successful video retrieval."""
        video_id = "video123"
        
        with patch.object(video_service, 'get_video') as mock_get:
            mock_get.return_value = {
                "id": "video123",
                "title": "Test Video",
                "owner_id": "user123",
            }
            
            result = await video_service.get_video(video_id)
            
            assert result["id"] == "video123"
            assert result["title"] == "Test Video"
            mock_get.assert_called_once_with(video_id)

    @pytest.mark.asyncio
    async def test_get_video_not_found(self, video_service):
        """Test video retrieval when video not found."""
        video_id = "nonexistent_video"
        
        with patch.object(video_service, 'get_video') as mock_get:
            mock_get.side_effect = VideoServiceError("Video not found")
            
            with pytest.raises(VideoServiceError):
                await video_service.get_video(video_id)

    @pytest.mark.asyncio
    async def test_update_video_success(self, video_service):
        """Test successful video update."""
        video_id = "video123"
        updates = {
            "title": "Updated Video Title",
            "description": "Updated description",
        }
        
        with patch.object(video_service, 'update_video') as mock_update:
            mock_update.return_value = {"status": "success"}
            
            result = await video_service.update_video(video_id, updates)
            
            assert result["status"] == "success"
            mock_update.assert_called_once_with(video_id, updates)

    @pytest.mark.asyncio
    async def test_delete_video_success(self, video_service):
        """Test successful video deletion."""
        video_id = "video123"
        
        with patch.object(video_service, 'delete_video') as mock_delete:
            mock_delete.return_value = {"status": "success"}
            
            result = await video_service.delete_video(video_id)
            
            assert result["status"] == "success"
            mock_delete.assert_called_once_with(video_id)

    @pytest.mark.asyncio
    async def test_like_video_success(self, video_service):
        """Test successful video like."""
        video_id = "video123"
        user_id = "user123"
        
        with patch.object(video_service, 'like_video') as mock_like:
            mock_like.return_value = {"status": "liked", "like_count": 1}
            
            result = await video_service.like_video(video_id, user_id)
            
            assert result["status"] == "liked"
            assert result["like_count"] == 1
            mock_like.assert_called_once_with(video_id, user_id)

    @pytest.mark.asyncio
    async def test_unlike_video_success(self, video_service):
        """Test successful video unlike."""
        video_id = "video123"
        user_id = "user123"
        
        with patch.object(video_service, 'unlike_video') as mock_unlike:
            mock_unlike.return_value = {"status": "unliked", "like_count": 0}
            
            result = await video_service.unlike_video(video_id, user_id)
            
            assert result["status"] == "unliked"
            assert result["like_count"] == 0
            mock_unlike.assert_called_once_with(video_id, user_id)

    @pytest.mark.asyncio
    async def test_view_video_success(self, video_service):
        """Test successful video view."""
        video_id = "video123"
        user_id = "user123"
        
        with patch.object(video_service, 'record_video_view') as mock_view:
            mock_view.return_value = {"status": "viewed", "view_count": 1}
            
            result = await video_service.view_video(video_id, user_id)
            
            assert result["status"] == "viewed"
            assert result["view_count"] == 1
            mock_view.assert_called_once_with(video_id, user_id)

    @pytest.mark.asyncio
    async def test_get_video_feed_success(self, video_service):
        """Test successful video feed retrieval."""
        user_id = "user123"
        limit = 20
        offset = 0
        
        with patch.object(video_service, 'get_video_feed') as mock_feed:
            mock_feed.return_value = {
                "videos": [
                    {"id": "video1", "title": "Video 1"},
                    {"id": "video2", "title": "Video 2"},
                ],
                "total": 2,
            }
            
            result = await video_service.get_video_feed(user_id, limit, offset)
            
            assert len(result["videos"]) == 2
            assert result["total"] == 2
            mock_feed.assert_called_once_with(user_id, limit, offset)

    @pytest.mark.asyncio
    async def test_search_videos_success(self, video_service):
        """Test successful video search."""
        query = "test video"
        filters = {"category": "gaming", "duration_min": 60}
        limit = 20
        offset = 0
        
        with patch.object(video_service, 'search_videos') as mock_search:
            mock_search.return_value = {
                "videos": [
                    {"id": "video1", "title": "Test Video 1"},
                    {"id": "video2", "title": "Test Video 2"},
                ],
                "total": 2,
            }
            
            result = await video_service.search_videos(query, filters, limit, offset)
            
            assert len(result["videos"]) == 2
            assert result["total"] == 2
            mock_search.assert_called_once_with(query, filters, limit, offset)

    @pytest.mark.asyncio
    async def test_get_video_analytics_success(self, video_service):
        """Test successful video analytics retrieval."""
        video_id = "video123"
        time_range = "30d"
        
        with patch.object(video_service, 'get_video_analytics') as mock_analytics:
            mock_analytics.return_value = {
                "video_id": "video123",
                "views": 1000,
                "likes": 50,
                "comments": 25,
                "shares": 10,
                "watch_time": 3600,
            }
            
            result = await video_service.get_video_analytics(video_id, time_range)
            
            assert result["video_id"] == "video123"
            assert result["views"] == 1000
            mock_analytics.assert_called_once_with(video_id, time_range)

    @pytest.mark.asyncio
    async def test_transcode_video_success(self, video_service):
        """Test successful video transcoding."""
        video_id = "video123"
        quality = "720p"
        
        with patch.object(video_service, 'transcode_video') as mock_transcode:
            mock_transcode.return_value = {
                "status": "transcoding",
                "job_id": "transcode_job_123",
            }
            
            result = await video_service.transcode_video(video_id, quality)
            
            assert result["status"] == "transcoding"
            assert result["job_id"] == "transcode_job_123"
            mock_transcode.assert_called_once_with(video_id, quality)

    @pytest.mark.asyncio
    async def test_generate_thumbnails_success(self, video_service):
        """Test successful thumbnail generation."""
        video_id = "video123"
        
        with patch.object(video_service, 'generate_thumbnails') as mock_thumbnails:
            mock_thumbnails.return_value = {
                "status": "success",
                "thumbnails": [
                    {"time": 10, "url": "thumbnail1.jpg"},
                    {"time": 30, "url": "thumbnail2.jpg"},
                ],
            }
            
            result = await video_service.generate_thumbnails(video_id)
            
            assert result["status"] == "success"
            assert len(result["thumbnails"]) == 2
            mock_thumbnails.assert_called_once_with(video_id)

    @pytest.mark.asyncio
    async def test_create_streaming_manifest_success(self, video_service):
        """Test successful streaming manifest creation."""
        video_id = "video123"
        
        with patch.object(video_service, 'create_streaming_manifest') as mock_manifest:
            mock_manifest.return_value = {
                "status": "success",
                "manifest_url": "https://example.com/manifest.m3u8",
            }
            
            result = await video_service.create_streaming_manifest(video_id)
            
            assert result["status"] == "success"
            assert "manifest_url" in result
            mock_manifest.assert_called_once_with(video_id)

    @pytest.mark.asyncio
    async def test_optimize_for_mobile_success(self, video_service):
        """Test successful mobile optimization."""
        video_id = "video123"
        
        with patch.object(video_service, 'optimize_for_mobile') as mock_optimize:
            mock_optimize.return_value = {
                "status": "success",
                "mobile_url": "https://example.com/mobile_video.mp4",
            }
            
            result = await video_service.optimize_for_mobile(video_id)
            
            assert result["status"] == "success"
            assert "mobile_url" in result
            mock_optimize.assert_called_once_with(video_id)
