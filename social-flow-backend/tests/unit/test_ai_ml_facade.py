import pytest

from app.ai_ml_services import get_ai_ml_service


@pytest.mark.asyncio
async def test_moderate_text_safe():
    ml = get_ai_ml_service()
    res = await ml.moderate_text("Hello wonderful community")
    assert res["is_safe"] is True


@pytest.mark.asyncio
async def test_moderate_text_flagged():
    ml = get_ai_ml_service()
    res = await ml.moderate_text("You are stupid idiot")
    assert res["flagged"] is True
    assert res["toxicity_score"] > 0


@pytest.mark.asyncio
async def test_generate_tags_deterministic():
    ml = get_ai_ml_service()
    t1 = await ml.generate_tags("Tech Advances", "AI tutorial and code examples")
    t2 = await ml.generate_tags("Tech Advances", "AI tutorial and code examples")
    assert t1 == t2
