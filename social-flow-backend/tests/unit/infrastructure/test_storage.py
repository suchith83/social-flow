"""
Unit Tests for Unified Storage Infrastructure.

Tests cover:
- S3Backend implementation
- StorageManager high-level API
- Multipart upload logic
- Error handling
"""

import pytest
import io
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.infrastructure.storage.base import StorageProvider, StorageMetadata
from app.infrastructure.storage.s3_backend import S3Backend
from app.infrastructure.storage.manager import StorageManager
from app.core.exceptions import StorageServiceError


class TestS3Backend:
    """Test S3Backend implementation."""

    @pytest.fixture
    def mock_aioboto3_session(self):
        """Mock aioboto3 session."""
        with patch('app.infrastructure.storage.s3_backend.aioboto3.Session') as mock:
            yield mock

    @pytest.fixture
    def s3_backend(self, mock_aioboto3_session):
        """Create S3Backend instance with mocked session."""
        with patch('app.infrastructure.storage.s3_backend.settings') as mock_settings:
            mock_settings.AWS_ACCESS_KEY_ID = "test_key"
            mock_settings.AWS_SECRET_ACCESS_KEY = "test_secret"
            mock_settings.AWS_REGION = "us-east-1"
            mock_settings.S3_BUCKET_NAME = "test-bucket"
            
            backend = S3Backend()
            return backend

    @pytest.mark.asyncio
    async def test_upload_success(self, s3_backend):
        """Test successful file upload."""
        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3.put_object = AsyncMock()
        mock_s3.head_object = AsyncMock(return_value={
            "ContentLength": 1024,
            "ContentType": "video/mp4",
            "ETag": "test-etag",
            "LastModified": datetime.now(),
            "Metadata": {}
        })
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        s3_backend.session.client = Mock(return_value=mock_context)
        
        # Test upload
        data = b"test video data"
        result = await s3_backend.upload(
            data=data,
            key="test/video.mp4",
            content_type="video/mp4"
        )
        
        assert isinstance(result, StorageMetadata)
        assert result.key == "test/video.mp4"
        assert result.size == 1024
        assert result.content_type == "video/mp4"
        assert result.provider == StorageProvider.S3
        
        # Verify S3 calls
        mock_s3.put_object.assert_called_once()
        mock_s3.head_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_success(self, s3_backend):
        """Test successful file download."""
        # Mock S3 client
        mock_stream = AsyncMock()
        mock_stream.read = AsyncMock(return_value=b"test data")
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)
        
        mock_s3 = AsyncMock()
        mock_s3.get_object = AsyncMock(return_value={"Body": mock_stream})
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        s3_backend.session.client = Mock(return_value=mock_context)
        
        # Test download
        data = await s3_backend.download(key="test/video.mp4")
        
        assert data == b"test data"
        mock_s3.get_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_success(self, s3_backend):
        """Test successful file deletion."""
        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3.delete_object = AsyncMock()
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        s3_backend.session.client = Mock(return_value=mock_context)
        
        # Test delete
        result = await s3_backend.delete(key="test/video.mp4")
        
        assert result is True
        mock_s3.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_exists_file_found(self, s3_backend):
        """Test file existence check - file exists."""
        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3.head_object = AsyncMock()
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        s3_backend.session.client = Mock(return_value=mock_context)
        
        # Test exists
        result = await s3_backend.exists(key="test/video.mp4")
        
        assert result is True
        mock_s3.head_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_presigned_url(self, s3_backend):
        """Test presigned URL generation."""
        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3.generate_presigned_url = AsyncMock(return_value="https://example.com/signed")
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        s3_backend.session.client = Mock(return_value=mock_context)
        
        # Test URL generation
        url = await s3_backend.generate_presigned_url(
            key="test/video.mp4",
            expires_in=3600
        )
        
        assert url == "https://example.com/signed"
        mock_s3.generate_presigned_url.assert_called_once()


class TestStorageManager:
    """Test StorageManager high-level API."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock storage backend."""
        backend = AsyncMock()
        backend.upload = AsyncMock(return_value=StorageMetadata(
            key="test/file.mp4",
            bucket="test-bucket",
            size=1024,
            content_type="video/mp4",
            provider=StorageProvider.S3,
            etag="test-etag"
        ))
        backend.download = AsyncMock(return_value=b"test data")
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=True)
        backend.generate_presigned_url = AsyncMock(return_value="https://example.com/signed")
        return backend

    @pytest.fixture
    def storage_manager(self, mock_backend):
        """Create StorageManager with mocked backend."""
        with patch('app.infrastructure.storage.manager.get_storage_manager') as mock:
            manager = StorageManager()
            manager._backends = {StorageProvider.S3: mock_backend}
            manager._active_provider = StorageProvider.S3
            mock.return_value = manager
            return manager

    @pytest.mark.asyncio
    async def test_upload_small_file(self, storage_manager, mock_backend):
        """Test upload of small file (no multipart)."""
        file_data = b"x" * 1024  # 1KB file
        file = io.BytesIO(file_data)
        
        result = await storage_manager.upload_file(
            file=file,
            key="test/small.mp4",
            content_type="video/mp4"
        )
        
        assert isinstance(result, StorageMetadata)
        assert result.key == "test/file.mp4"
        mock_backend.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_large_file_triggers_multipart(self, storage_manager, mock_backend):
        """Test upload of large file triggers multipart."""
        # Create file larger than threshold (5MB)
        file_data = b"x" * (6 * 1024 * 1024)  # 6MB
        file = io.BytesIO(file_data)
        
        # Mock multipart methods
        mock_backend.initiate_multipart_upload = AsyncMock(return_value="test-upload-id")
        mock_backend.upload_part = AsyncMock(return_value={"PartNumber": 1, "ETag": "etag-1"})
        mock_backend.complete_multipart_upload = AsyncMock(return_value=StorageMetadata(
            key="test/large.mp4",
            bucket="test-bucket",
            size=6 * 1024 * 1024,
            content_type="video/mp4",
            provider=StorageProvider.S3
        ))
        
        result = await storage_manager.upload_large_file(
            file=file,
            key="test/large.mp4",
            content_type="video/mp4"
        )
        
        assert isinstance(result, StorageMetadata)
        mock_backend.initiate_multipart_upload.assert_called_once()
        assert mock_backend.upload_part.call_count > 0
        mock_backend.complete_multipart_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file(self, storage_manager, mock_backend):
        """Test file download."""
        data = await storage_manager.download_file(key="test/file.mp4")
        
        assert data == b"test data"
        mock_backend.download.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_file(self, storage_manager, mock_backend):
        """Test file deletion."""
        result = await storage_manager.delete_file(key="test/file.mp4")
        
        assert result is True
        mock_backend.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_public_url(self, storage_manager, mock_backend):
        """Test presigned URL generation."""
        url = await storage_manager.get_public_url(
            key="test/file.mp4",
            expires_in=3600
        )
        
        assert url == "https://example.com/signed"
        mock_backend.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_exists(self, storage_manager, mock_backend):
        """Test file existence check."""
        exists = await storage_manager.file_exists(key="test/file.mp4")
        
        assert exists is True
        mock_backend.exists.assert_called_once()

    def test_switch_provider(self, storage_manager):
        """Test provider switching."""
        # Add another provider
        mock_azure = AsyncMock()
        storage_manager._backends[StorageProvider.AZURE] = mock_azure
        
        # Switch provider
        storage_manager.switch_provider(StorageProvider.AZURE)
        
        assert storage_manager._active_provider == StorageProvider.AZURE

    def test_switch_to_unavailable_provider_fails(self, storage_manager):
        """Test switching to unavailable provider raises error."""
        with pytest.raises(StorageServiceError):
            storage_manager.switch_provider(StorageProvider.AZURE)

    @pytest.mark.asyncio
    async def test_content_type_detection(self, storage_manager, mock_backend):
        """Test automatic content type detection."""
        file = io.BytesIO(b"test")
        
        await storage_manager.upload_file(
            file=file,
            key="test/video.mp4",
            content_type=None  # Should auto-detect
        )
        
        # Verify upload was called (content type should be detected)
        mock_backend.upload.assert_called_once()
        call_args = mock_backend.upload.call_args
        assert "content_type" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_multipart_abort_on_error(self, storage_manager, mock_backend):
        """Test multipart upload aborts on error."""
        file_data = b"x" * (6 * 1024 * 1024)
        file = io.BytesIO(file_data)
        
        # Mock multipart methods with error
        mock_backend.initiate_multipart_upload = AsyncMock(return_value="test-upload-id")
        mock_backend.upload_part = AsyncMock(side_effect=Exception("Upload failed"))
        mock_backend.abort_multipart_upload = AsyncMock(return_value=True)
        
        with pytest.raises(StorageServiceError):
            await storage_manager.upload_large_file(
                file=file,
                key="test/large.mp4"
            )
        
        # Verify abort was called
        mock_backend.abort_multipart_upload.assert_called_once()


class TestLegacyWrapper:
    """Test legacy storage service wrapper."""

    @pytest.mark.asyncio
    async def test_legacy_upload_file(self):
        """Test legacy upload_file method."""
        from app.services.storage_service import storage_service
        
        with patch.object(storage_service._manager, 'upload_file') as mock_upload:
            with patch.object(storage_service._manager, 'get_public_url') as mock_url:
                mock_upload.return_value = StorageMetadata(
                    key="test/file.mp4",
                    bucket="test-bucket",
                    size=1024,
                    content_type="video/mp4",
                    provider=StorageProvider.S3,
                    created_at=datetime.now()
                )
                mock_url.return_value = "https://example.com/file.mp4"
                
                result = await storage_service.upload_file(
                    file_data=b"test",
                    file_path="test/file.mp4",
                    content_type="video/mp4"
                )
                
                assert "file_path" in result
                assert "url" in result
                assert result["file_path"] == "test/file.mp4"

    @pytest.mark.asyncio
    async def test_legacy_download_file(self):
        """Test legacy download_file method."""
        from app.services.storage_service import storage_service
        
        with patch.object(storage_service._manager, 'download_file') as mock_download:
            mock_download.return_value = b"test data"
            
            data = await storage_service.download_file(file_path="test/file.mp4")
            
            assert data == b"test data"

    @pytest.mark.asyncio
    async def test_legacy_delete_file(self):
        """Test legacy delete_file method."""
        from app.services.storage_service import storage_service
        
        with patch.object(storage_service._manager, 'delete_file') as mock_delete:
            mock_delete.return_value = True
            
            result = await storage_service.delete_file(file_path="test/file.mp4")
            
            assert "status" in result
            assert result["status"] == "deleted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
