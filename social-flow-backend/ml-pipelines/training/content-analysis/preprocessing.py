# Clean and preprocess text or other inputs
# ============================
# File: preprocessing.py
# ============================
import re
from typing import List

def clean_text(text: str) -> str:
    """Basic text preprocessing: lowercasing, removing URLs, mentions, etc."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove urls
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(r"[^a-z0-9\s]", "", text)  # keep only alphanumerics
    return text.strip()

def preprocess_batch(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]
