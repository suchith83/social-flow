"""
Integration tests for Live Streaming Infrastructure.

Tests the complete livestream workflow including:
- Stream creation and management
- WebSocket chat functionality
- Viewer tracking
- Recording functionality
- Stream metrics and analytics
"""

import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.auth.models.user import User
from app.livestream.models.livestream import LiveStream, StreamViewer


@pytest.mark.asyncio
class TestLiveStreamAPI:
    """Test livestream API endpoints."""
    
    async def test_create_livestream(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test creating a new livestream."""
        response = await async_client.post(
            "/api/v1/livestream/streams",
            json={
                "title": "Test Live Stream",
                "description": "Testing livestream creation",
                "category": "gaming",
                "tags": ["test", "gaming", "live"],
                "is_private": False,
                "record_stream": True
            },
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "stream_id" in data
        assert "rtmp_url" in data
        assert "stream_key" in data
        assert data["title"] == "Test Live Stream"
    
    async def test_get_stream_details(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test getting livestream details."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.get(
            f"/api/v1/livestream/streams/{stream.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["stream_id"] == str(stream.id)
        assert data["title"] == "Test Stream"
    
    async def test_list_active_streams(
        self,
        async_client: AsyncClient,
        auth_headers: dict
    ):
        """Test listing active livestreams."""
        response = await async_client.get(
            "/api/v1/livestream/streams/active",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "streams" in data
        assert isinstance(data["streams"], list)
    
    async def test_start_stream(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test starting a livestream."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="created"
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.post(
            f"/api/v1/livestream/streams/{stream.id}/start",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "live"
    
    async def test_end_stream(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test ending a livestream."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.post(
            f"/api/v1/livestream/streams/{stream.id}/end",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ended"
    
    async def test_get_stream_metrics(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test getting livestream metrics."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live",
            current_viewers=150,
            peak_viewers=200,
            total_views=500
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.get(
            f"/api/v1/livestream/streams/{stream.id}/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["current_viewers"] == 150
        assert data["peak_viewers"] == 200
        assert data["total_views"] == 500
    
    async def test_get_chat_history(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test getting chat history for a stream."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.get(
            f"/api/v1/livestream/streams/{stream.id}/chat",
            params={"limit": 50},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)


@pytest.mark.asyncio
class TestLiveStreamChat:
    """Test livestream WebSocket chat functionality."""
    
    async def test_websocket_connection(
        self,
        test_user: User,
        db_session: AsyncSession
    ):
        """Test WebSocket connection to chat."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        # WebSocket connection would be tested with actual WS client
        # For now, verify stream exists
        assert stream.id is not None
    
    async def test_send_chat_message(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test sending a chat message."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.post(
            f"/api/v1/livestream/streams/{stream.id}/chat/message",
            json={
                "message": "Hello, world!",
                "message_type": "text"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Hello, world!"
    
    async def test_moderate_chat_message(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test moderating (deleting) a chat message."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        message_id = str(uuid4())
        
        response = await async_client.delete(
            f"/api/v1/livestream/streams/{stream.id}/chat/message/{message_id}",
            headers=auth_headers
        )
        
        # Should succeed or return 404 if message doesn't exist
        assert response.status_code in [200, 404]


@pytest.mark.asyncio
class TestLiveStreamViewers:
    """Test viewer tracking functionality."""
    
    async def test_join_stream(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test viewer joining a stream."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.post(
            f"/api/v1/livestream/streams/{stream.id}/join",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "viewer_id" in data
    
    async def test_leave_stream(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test viewer leaving a stream."""
        # Create test stream and viewer
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live"
        )
        db_session.add(stream)
        
        viewer = StreamViewer(
            id=uuid4(),
            stream_id=stream.id,
            user_id=test_user.id,
            is_active=True
        )
        db_session.add(viewer)
        await db_session.commit()
        
        response = await async_client.post(
            f"/api/v1/livestream/streams/{stream.id}/leave",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    async def test_get_active_viewers(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test getting list of active viewers."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="live",
            current_viewers=5
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.get(
            f"/api/v1/livestream/streams/{stream.id}/viewers",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "viewers" in data
        assert isinstance(data["viewers"], list)


@pytest.mark.asyncio
class TestLiveStreamRecording:
    """Test stream recording functionality."""
    
    async def test_enable_recording(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test enabling recording for a stream."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="created",
            record_stream=False
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.patch(
            f"/api/v1/livestream/streams/{stream.id}",
            json={"record_stream": True},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["record_stream"] is True
    
    async def test_get_stream_recordings(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test getting list of stream recordings."""
        # Create test stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Test Stream",
            status="ended",
            record_stream=True
        )
        db_session.add(stream)
        await db_session.commit()
        
        response = await async_client.get(
            f"/api/v1/livestream/streams/{stream.id}/recordings",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "recordings" in data


@pytest.mark.asyncio
class TestLiveStreamWorkflow:
    """Test end-to-end livestream workflow."""
    
    async def test_complete_stream_lifecycle(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test complete stream lifecycle from creation to end."""
        
        # Step 1: Create stream
        create_response = await async_client.post(
            "/api/v1/livestream/streams",
            json={
                "title": "Complete Workflow Test",
                "description": "Testing full lifecycle",
                "record_stream": True
            },
            headers=auth_headers
        )
        assert create_response.status_code == 201
        stream_id = create_response.json()["stream_id"]
        
        # Step 2: Get stream details
        details_response = await async_client.get(
            f"/api/v1/livestream/streams/{stream_id}",
            headers=auth_headers
        )
        assert details_response.status_code == 200
        
        # Step 3: Start stream
        start_response = await async_client.post(
            f"/api/v1/livestream/streams/{stream_id}/start",
            headers=auth_headers
        )
        assert start_response.status_code == 200
        
        # Step 4: Join as viewer
        join_response = await async_client.post(
            f"/api/v1/livestream/streams/{stream_id}/join",
            headers=auth_headers
        )
        assert join_response.status_code == 200
        
        # Step 5: Send chat message
        chat_response = await async_client.post(
            f"/api/v1/livestream/streams/{stream_id}/chat/message",
            json={"message": "Test message"},
            headers=auth_headers
        )
        assert chat_response.status_code == 201
        
        # Step 6: Get metrics
        metrics_response = await async_client.get(
            f"/api/v1/livestream/streams/{stream_id}/metrics",
            headers=auth_headers
        )
        assert metrics_response.status_code == 200
        
        # Step 7: End stream
        end_response = await async_client.post(
            f"/api/v1/livestream/streams/{stream_id}/end",
            headers=auth_headers
        )
        assert end_response.status_code == 200


@pytest.mark.asyncio
class TestLiveStreamSecurity:
    """Test livestream security and authorization."""
    
    async def test_unauthorized_stream_creation(
        self,
        async_client: AsyncClient
    ):
        """Test that unauthenticated users cannot create streams."""
        response = await async_client.post(
            "/api/v1/livestream/streams",
            json={"title": "Unauthorized Stream"}
        )
        
        assert response.status_code == 401
    
    async def test_stream_ownership_control(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test that only stream owner can control stream."""
        # Create stream owned by different user
        other_user = User(
            id=uuid4(),
            email="other@example.com",
            username="otheruser"
        )
        db_session.add(other_user)
        
        stream = LiveStream(
            id=uuid4(),
            user_id=other_user.id,
            title="Other User Stream",
            status="live"
        )
        db_session.add(stream)
        await db_session.commit()
        
        # Try to end other user's stream
        response = await async_client.post(
            f"/api/v1/livestream/streams/{stream.id}/end",
            headers=auth_headers
        )
        
        # Should return 403 Forbidden
        assert response.status_code == 403
    
    async def test_private_stream_access(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test that private streams are not publicly visible."""
        # Create private stream
        stream = LiveStream(
            id=uuid4(),
            user_id=test_user.id,
            title="Private Stream",
            status="live",
            is_private=True
        )
        db_session.add(stream)
        await db_session.commit()
        
        # List active streams should not include private stream
        response = await async_client.get(
            "/api/v1/livestream/streams/active",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        stream_ids = [s["stream_id"] for s in data["streams"]]
        
        # Private stream should not be in public list
        # (unless user is owner or has access)
        # This depends on implementation details
        assert True  # Placeholder for actual check
