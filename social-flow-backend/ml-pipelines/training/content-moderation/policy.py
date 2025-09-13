# Policy definitions for moderation
# ============================
# File: policy.py
# ============================
"""
Policy engine integrates ML + rules to produce final moderation decisions.
"""
from typing import List

class ModerationPolicy:
    def __init__(self, labels: List[str]):
        self.labels = labels

    def decide(self, ml_prediction: int, rule_flags: List[str]) -> str:
        if "nsfw" in rule_flags:
            return "nsfw"
        if "profanity" in rule_flags and self.labels[ml_prediction] != "safe":
            return "profanity"
        return self.labels[ml_prediction]
