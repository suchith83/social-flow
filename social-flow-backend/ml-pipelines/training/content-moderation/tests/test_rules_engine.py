# Unit tests for rules_engine.py
# ============================
# File: tests/test_rules_engine.py
# ============================
from ml_pipelines.training.content_moderation.rules_engine import ModerationRules

def test_rules_detection(tmp_path):
    profanity_file = tmp_path / "profanity.txt"
    profanity_file.write_text("badword\n")
    rules = ModerationRules(str(profanity_file), ["xxx"], 0.5)
    assert "profanity" in rules.apply_rules("this is a badword")
    assert "nsfw" in rules.apply_rules("watch xxx now")
    assert "spam" in rules.apply_rules("http://a.com http://b.com")
