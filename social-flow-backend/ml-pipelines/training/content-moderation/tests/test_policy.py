# Unit tests for policy.py
# ============================
# File: tests/test_policy.py
# ============================
from ml_pipelines.training.content_moderation.policy import ModerationPolicy

def test_policy_priority():
    policy = ModerationPolicy(["safe", "spam", "hate", "nsfw", "harassment"])
    assert policy.decide(0, ["nsfw"]) == "nsfw"
    assert policy.decide(2, ["profanity"]) == "profanity"
    assert policy.decide(1, []) == "spam"
