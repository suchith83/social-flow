import pytest
import asyncio

class VideoService:
    def __init__(self, db, s3_client):
        self.db = db
        self.s3 = s3_client

    async def upload_video(self, video_id, content: bytes):
        self.db["videos"][video_id] = {"status": "uploading"}
        await asyncio.sleep(0.01)  # simulate async
        self.s3.upload_file("bucket", video_id, content)
        self.db["videos"][video_id]["status"] = "uploaded"
        return True

@pytest.mark.asyncio
async def test_upload_video(fake_db, mock_s3_client):
    service = VideoService(fake_db, mock_s3_client)
    result = await service.upload_video("v1", b"binary-content")
    assert result is True
    assert fake_db["videos"]["v1"]["status"] == "uploaded"
    mock_s3_client.upload_file.assert_called_once()
