"""
Caption Generation Package
Supports image captioning using CNN encoders + Transformer decoders.
"""

__version__ = "1.0.0"
__author__ = "AI Caption Generator"

"""Caption generation local stub."""
from typing import Any, Dict


def load_model(config: dict = None):
    class CaptionGen:
        def predict(self, video_or_image: Any) -> Dict[str, Any]:
            return {"caption": "A generated caption for the content."}

    return CaptionGen()
