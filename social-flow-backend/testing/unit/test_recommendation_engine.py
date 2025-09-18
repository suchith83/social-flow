import pytest
import random
from hypothesis import given, strategies as st

class RecommendationEngine:
    def recommend(self, user_id, history):
        """
        Dummy recommender: shuffle history and return top-3.
        """
        if not history:
            return []
        recs = list(history)
        random.shuffle(recs)
        return recs[:3]

@given(st.lists(st.text(), min_size=0, max_size=20))
def test_recommendations_do_not_exceed_three(history):
    engine = RecommendationEngine()
    recs = engine.recommend("u1", history)
    assert len(recs) <= 3

def test_recommend_empty_history():
    engine = RecommendationEngine()
    assert engine.recommend("u1", []) == []
