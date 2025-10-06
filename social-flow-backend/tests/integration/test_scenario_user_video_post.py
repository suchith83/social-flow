from app.auth.schemas.auth import UserCreate
from app.videos.schemas.video import VideoCreate
from app.posts.schemas.post import PostCreate
from app.services.recommendation_service import RecommendationService
import pytest

@pytest.mark.asyncio
async def test_scenario_user_video_post_and_recommendations(db_session):
    # Validate schemas (password must include upper, lower, digit per validator)
    user = UserCreate(email="user@example.com", username="user1", password="SuperSecret1")
    video = VideoCreate(title="Demo", description="Test video", tags=["demo"], duration=10, file_size=1000, original_filename="demo.mp4")
    post = PostCreate(content="Great content here!", visibility="public")
    assert user.email and video.title and post.content
    # Invoke recommendations (no real data persistence needed for heuristic response)
    rec_service = RecommendationService.from_container(db_session)
    recs = await rec_service.get_video_recommendations(user_id=None, limit=2, algorithm="hybrid")
    assert "recommendations" in recs