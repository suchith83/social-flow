# Text cleaning and preprocessing
# ============================
# File: preprocessing.py
# ============================
import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#\w+", "", text)
    return text.strip()
