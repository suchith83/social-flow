"""
Test configuration and fixtures.

This module contains pytest configuration and shared fixtures
for testing the Social Flow backend.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient

from app.main import app
from app.core.database import get_db, Base
from app.core.config import settings
from app.models import User, Video, Post, Comment, Like, Follow, Ad, Payment, Subscription, Notification, Analytics, ViewCount, LiveStream, LiveStreamViewer


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
)

# Create test session factory
TestSessionLocal = sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestSessionLocal() as session:
        yield session
    
    # Drop tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
def client(db_session: AsyncSession) -> Generator[TestClient, None, None]:
    """Create a test client."""
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None, None]:
    """Create an async test client."""
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        display_name="Test User",
        bio="Test bio",
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture(scope="function")
async def test_video(db_session: AsyncSession, test_user: User) -> Video:
    """Create a test video."""
    video = Video(
        title="Test Video",
        description="Test video description",
        filename="test_video.mp4",
        file_size=1024000,
        duration=120.0,
        resolution="1920x1080",
        bitrate=2000,
        codec="h264",
        s3_key="videos/test_video.mp4",
        s3_bucket="test-bucket",
        owner_id=test_user.id,
    )
    db_session.add(video)
    await db_session.commit()
    await db_session.refresh(video)
    return video


@pytest_asyncio.fixture(scope="function")
async def test_post(db_session: AsyncSession, test_user: User) -> Post:
    """Create a test post."""
    post = Post(
        content="Test post content",
        owner_id=test_user.id,
    )
    db_session.add(post)
    await db_session.commit()
    await db_session.refresh(post)
    return post


@pytest_asyncio.fixture(scope="function")
async def test_comment(db_session: AsyncSession, test_user: User, test_video: Video) -> Comment:
    """Create a test comment."""
    comment = Comment(
        content="Test comment",
        owner_id=test_user.id,
        video_id=test_video.id,
    )
    db_session.add(comment)
    await db_session.commit()
    await db_session.refresh(comment)
    return comment


@pytest_asyncio.fixture(scope="function")
async def test_like(db_session: AsyncSession, test_user: User, test_video: Video) -> Like:
    """Create a test like."""
    like = Like(
        user_id=test_user.id,
        video_id=test_video.id,
        is_like=True,
    )
    db_session.add(like)
    await db_session.commit()
    await db_session.refresh(like)
    return like


@pytest_asyncio.fixture(scope="function")
async def test_follow(db_session: AsyncSession, test_user: User) -> Follow:
    """Create a test follow relationship."""
    # Create another user to follow
    followed_user = User(
        username="followeduser",
        email="followed@example.com",
        password_hash="hashed_password",
        display_name="Followed User",
        is_active=True,
    )
    db_session.add(followed_user)
    await db_session.commit()
    await db_session.refresh(followed_user)
    
    follow = Follow(
        follower_id=test_user.id,
        following_id=followed_user.id,
    )
    db_session.add(follow)
    await db_session.commit()
    await db_session.refresh(follow)
    return follow


@pytest_asyncio.fixture(scope="function")
async def test_ad(db_session: AsyncSession) -> Ad:
    """Create a test ad."""
    ad = Ad(
        title="Test Ad",
        description="Test ad description",
        ad_type="pre-roll",
        duration=15,
        url="https://example.com/ad.mp4",
        click_url="https://example.com/click",
        impression_url="https://example.com/impression",
        is_active=True,
    )
    db_session.add(ad)
    await db_session.commit()
    await db_session.refresh(ad)
    return ad


@pytest_asyncio.fixture(scope="function")
async def test_payment(db_session: AsyncSession, test_user: User) -> Payment:
    """Create a test payment."""
    payment = Payment(
        user_id=test_user.id,
        amount=1000,  # $10.00 in cents
        currency="USD",
        payment_method="stripe",
        status="completed",
        transaction_id="test_transaction_123",
    )
    db_session.add(payment)
    await db_session.commit()
    await db_session.refresh(payment)
    return payment


@pytest_asyncio.fixture(scope="function")
async def test_subscription(db_session: AsyncSession, test_user: User) -> Subscription:
    """Create a test subscription."""
    subscription = Subscription(
        user_id=test_user.id,
        plan="premium",
        status="active",
        amount=999,  # $9.99 in cents
        currency="USD",
        billing_cycle="monthly",
    )
    db_session.add(subscription)
    await db_session.commit()
    await db_session.refresh(subscription)
    return subscription


@pytest_asyncio.fixture(scope="function")
async def test_notification(db_session: AsyncSession, test_user: User) -> Notification:
    """Create a test notification."""
    notification = Notification(
        user_id=test_user.id,
        title="Test Notification",
        message="Test notification message",
        notification_type="general",
        is_read=False,
    )
    db_session.add(notification)
    await db_session.commit()
    await db_session.refresh(notification)
    return notification


@pytest_asyncio.fixture(scope="function")
async def test_analytics(db_session: AsyncSession, test_user: User) -> Analytics:
    """Create a test analytics event."""
    analytics = Analytics(
        event_type="video_view",
        category="content",
        event="video_viewed",
        entity_type="video",
        entity_id="test_video_id",
        user_id=test_user.id,
        properties='{"video_id": "test_video_id", "duration": 120}',
    )
    db_session.add(analytics)
    await db_session.commit()
    await db_session.refresh(analytics)
    return analytics


@pytest_asyncio.fixture(scope="function")
async def test_view_count(db_session: AsyncSession, test_video: Video) -> ViewCount:
    """Create a test view count."""
    view_count = ViewCount(
        video_id=test_video.id,
        count=100,
        date="2025-01-01",
    )
    db_session.add(view_count)
    await db_session.commit()
    await db_session.refresh(view_count)
    return view_count


@pytest_asyncio.fixture(scope="function")
async def test_live_stream(db_session: AsyncSession, test_user: User) -> LiveStream:
    """Create a test live stream."""
    live_stream = LiveStream(
        title="Test Live Stream",
        description="Test live stream description",
        stream_key="test_stream_key_123",
        channel_arn="arn:aws:ivs:us-west-2:123456789012:channel/test_channel",
        ingest_endpoint="rtmp://test.ingest.ivs.amazonaws.com/live/test_stream_key_123",
        playback_url="https://test.playback.ivs.amazonaws.com/live/test_channel.m3u8",
        owner_id=test_user.id,
    )
    db_session.add(live_stream)
    await db_session.commit()
    await db_session.refresh(live_stream)
    return live_stream


@pytest_asyncio.fixture(scope="function")
async def test_live_stream_viewer(db_session: AsyncSession, test_live_stream: LiveStream, test_user: User) -> LiveStreamViewer:
    """Create a test live stream viewer."""
    viewer = LiveStreamViewer(
        live_stream_id=test_live_stream.id,
        user_id=test_user.id,
        session_id="test_session_123",
        watch_duration=300,  # 5 minutes
    )
    db_session.add(viewer)
    await db_session.commit()
    await db_session.refresh(viewer)
    return viewer


@pytest.fixture(scope="function")
def auth_headers(test_user: User) -> dict:
    """Create authentication headers for test user."""
    # TODO: Generate actual JWT token for test user
    return {"Authorization": f"Bearer test_token_{test_user.id}"}


@pytest.fixture(scope="function")
def test_data() -> dict:
    """Create test data dictionary."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test User",
        "bio": "Test bio",
    }


@pytest.fixture(scope="function")
def video_data() -> dict:
    """Create test video data dictionary."""
    return {
        "title": "Test Video",
        "description": "Test video description",
        "tags": ["test", "video"],
        "visibility": "public",
    }


@pytest.fixture(scope="function")
def post_data() -> dict:
    """Create test post data dictionary."""
    return {
        "content": "Test post content",
        "visibility": "public",
    }


@pytest.fixture(scope="function")
def comment_data() -> dict:
    """Create test comment data dictionary."""
    return {
        "content": "Test comment",
    }


@pytest.fixture(scope="function")
def live_stream_data() -> dict:
    """Create test live stream data dictionary."""
    return {
        "title": "Test Live Stream",
        "description": "Test live stream description",
        "tags": ["test", "live"],
        "visibility": "public",
    }
