"""
Sprite/contact-sheet generation.

Generates a single image that is a grid of thumbnails, plus a metadata JSON describing
coordinates (useful for sprite+vtt setups).
"""

import os
from PIL import Image
from typing import List, Tuple, Dict
from .utils import ensure_dir, logger
from .models import ThumbnailResult
from .config import config


def generate_contact_sheet(thumbnail_paths: List[str], output_path: str, cols: int = 4, thumb_size: Tuple[int,int]=None) -> Dict:
    """
    Create a contact sheet image from list of thumbnail file paths.
    Returns metadata dictionary with coordinates and sprite info.
    """
    if not thumbnail_paths:
        raise ValueError("No thumbnails provided")
    images = [Image.open(p).convert("RGB") for p in thumbnail_paths]
    # optionally resize to same size
    if thumb_size:
        images = [img.resize(thumb_size, Image.LANCZOS) for img in images]
    widths, heights = zip(*(i.size for i in images))
    tw = max(widths)
    th = max(heights)
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    sheet_w = cols * tw
    sheet_h = rows * th
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0,0,0))
    meta = {"sprite": os.path.basename(output_path), "thumb_w": tw, "thumb_h": th, "cols": cols, "rows": rows, "frames": []}
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * tw
        y = r * th
        sheet.paste(img, (x,y))
        meta["frames"].append({"index": idx, "x": x, "y": y, "w": img.size[0], "h": img.size[1], "path": os.path.basename(thumbnail_paths[idx])})
    ensure_dir(os.path.dirname(output_path))
    sheet.save(output_path, "JPEG", quality=85)
    logger.info("Generated contact sheet at %s", output_path)
    return meta
