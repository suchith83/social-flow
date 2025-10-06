import pytest

from app.services.recommendation_service import RecommendationService


@pytest.mark.asyncio
async def test_recommendations_hybrid(db_session):
    svc = RecommendationService.from_container(db_session)
    res = await svc.get_video_recommendations(user_id=None, limit=5, algorithm="hybrid")
    assert "recommendations" in res


@pytest.mark.asyncio
async def test_recommendations_trending(db_session):
    svc = RecommendationService.from_container(db_session)
    res = await svc.get_video_recommendations(user_id=None, limit=3, algorithm="trending")
    assert len(res.get("recommendations", [])) <= 3
