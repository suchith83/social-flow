import pytest
from app.ml.services.ml_service import MLService, RecommendationAlgorithm

@pytest.mark.asyncio
async def test_moderation_aggregation_deterministic():
    svc = MLService()
    r1 = await svc.aggregate_moderation(text="BUY now!!! Limited time offer!!!")
    r2 = await svc.aggregate_moderation(text="BUY now!!! Limited time offer!!!")
    assert r1["success"] is True
    assert r1["data"]["risk_score"] == r2["data"]["risk_score"]
    assert set(r1["data"].keys()) >= {"scores","risk_score","flags","is_safe"}

@pytest.mark.asyncio
async def test_emotional_profile_intents():
    svc = MLService()
    promo = await svc.emotional_profile("Buy this amazing product now!")
    question = await svc.emotional_profile("What is the best way to learn Python?")
    help_req = await svc.emotional_profile("How do I reset my password")
    assert promo["data"]["intent"] == "promote"
    assert question["data"]["intent"] == "question"
    assert help_req["data"]["intent"] == "seek_help"

@pytest.mark.asyncio
async def test_video_composite_analysis_caching():
    svc = MLService()
    first = await svc.video_composite_analysis("video123")
    second = await svc.video_composite_analysis("video123")
    assert first["meta"].get("cached") is False
    assert second["meta"].get("cached") is True

@pytest.mark.asyncio
async def test_duplicate_detection_flow():
    svc = MLService()
    first = await svc.detect_duplicates("content_abc")
    second = await svc.detect_duplicates("content_abc")
    assert first["is_duplicate"] is False
    assert second["is_duplicate"] is True
    assert len(second["similar_content"]) >= 1

@pytest.mark.asyncio
async def test_viral_potential_caching():
    svc = MLService()
    v1 = await svc.viral_potential("vid42", creator_followers=50000, recent_engagement={"views":1000,"likes":200,"comments":50,"shares":25})
    v2 = await svc.viral_potential("vid42", creator_followers=50000, recent_engagement={"views":1000,"likes":200,"comments":50,"shares":25})
    assert v1["data"]["viral_score"] == v2["data"]["viral_score"]
    assert v1["meta"].get("cached") is False
    assert v2["meta"].get("cached") is True

@pytest.mark.asyncio
async def test_route_recommendations_basic():
    svc = MLService()
    result = await svc.route_recommendations("user_z", limit=5, algorithm=RecommendationAlgorithm.TRENDING)
    assert result["success"] is True
    assert result["meta"]["algorithm"] == RecommendationAlgorithm.TRENDING.value
