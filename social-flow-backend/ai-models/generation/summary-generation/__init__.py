"""
Summary Generation Package
Supports extractive and abstractive summarization, training and inference.
"""

__version__ = "1.0.0"
__author__ = "AI Summarizer"

"""Summary generation local stub."""
from typing import Any, Dict


def load_model(config: dict = None):
    class SummaryGen:
        def predict(self, text: str, max_length: int = 120) -> Dict[str, Any]:
            # Trivial summarization stub: return first 120 chars
            summary = (text or "")[:max_length]
            return {"summary": summary}

    return SummaryGen()
