# Unit tests for preprocessing
# ============================
# File: tests/test_preprocessing.py
# ============================
from ml_pipelines.training.content_analysis.preprocessing import clean_text

def test_clean_text():
    text = "Check this out! http://example.com @user #hashtag"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
