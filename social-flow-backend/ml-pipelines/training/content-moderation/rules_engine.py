# Rule-based moderation logic
# ============================
# File: rules_engine.py
# ============================
import re
from typing import List

class ModerationRules:
    """Rule-based moderation system with regex + keyword matching."""

    def __init__(self, profanity_list: str, nsfw_keywords: List[str], spam_threshold: float):
        self.profanities = set(open(profanity_list).read().splitlines()) if profanity_list else set()
        self.nsfw_keywords = nsfw_keywords
        self.spam_threshold = spam_threshold

    def check_profanity(self, text: str) -> bool:
        return any(word in text.lower().split() for word in self.profanities)

    def check_nsfw(self, text: str) -> bool:
        return any(keyword in text.lower() for keyword in self.nsfw_keywords)

    def check_spam(self, text: str) -> bool:
        links = len(re.findall(r"http[s]?://", text))
        words = len(text.split())
        return links / max(words, 1) > self.spam_threshold

    def apply_rules(self, text: str) -> List[str]:
        flags = []
        if self.check_profanity(text):
            flags.append("profanity")
        if self.check_nsfw(text):
            flags.append("nsfw")
        if self.check_spam(text):
            flags.append("spam")
        return flags
