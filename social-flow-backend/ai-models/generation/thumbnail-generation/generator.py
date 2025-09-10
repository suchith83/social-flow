"""
High-level inference utilities for generating thumbnails from a source image or video frame.

Features:
 - multi-aspect outputs
 - smart crop guidance: return bounding boxes with top scoring crops (based on generator + optional scoring fn)
 - batching & fast resizing
"""

from PIL import Image
import torch
from typing import List, Tuple
from .models import ThumbnailGenerator
from .utils import pil_to_tensor, tensor_to_pil, get_device
import torchvision.transforms.functional as TF
import numpy as np

import logging
logger = logging.getLogger("thumbnail_generation.generator")

class ThumbnailService:
    def __init__(self, generator: ThumbnailGenerator, device: str = "cuda"):
        self.device = get_device(device)
        self.gen = generator.to(self.device)
        self.gen.eval()

    def generate_thumbnails(self, pil_img: Image.Image, sizes: List[Tuple[int,int]] = None) -> List[Tuple[Tuple[int,int], Image.Image]]:
        """
        Generate thumbnails at requested sizes.
        Returns list of (size, PIL Image)
        """
        sizes = sizes or [(320,180)]
        # prepare source image at src_size expected by model
        src = pil_img.convert("RGB")
        src_t = pil_to_tensor(src).unsqueeze(0).to(self.device)  # 1xCxHxW (already assumed resized in dataset pipeline)
        with torch.no_grad():
            gen_full = self.gen(src_t)  # 1x3xH'xW' where H',W' ~ src_size/ (maybe)
            out = []
            for size in sizes:
                w,h = size
                thumb = torch.nn.functional.interpolate(gen_full, size=(h,w), mode="bilinear", align_corners=False)
                thumb_pil = tensor_to_pil(thumb[0])
                out.append((size, thumb_pil))
        return out

    def smart_crop_suggestions(self, pil_img: Image.Image, candidate_grid: Tuple[int,int] = (5,5), crop_size: Tuple[int,int] = (320,180), top_k: int = 3):
        """
        Brute-force candidate crops arranged in grid; score crops using generator-based scoring:
         - pass crop through generator (from full-res source) and compute a quality score (simple heuristic: high contrast / high detail measured by Laplacian variance)
         - sort and return top_k crop boxes and cropped images
        This is a heuristic helper (fast, no learned scorer). Replace with learned scorer for production.
        """
        W, H = pil_img.size
        grid_x, grid_y = candidate_grid
        step_x = max(1, (W - crop_size[0]) // (grid_x - 1)) if grid_x>1 else max(1, W - crop_size[0])
        step_y = max(1, (H - crop_size[1]) // (grid_y - 1)) if grid_y>1 else max(1, H - crop_size[1])

        candidates = []
        for gx in range(grid_x):
            for gy in range(grid_y):
                x = min(max(0, gx * step_x), W - crop_size[0])
                y = min(max(0, gy * step_y), H - crop_size[1])
                crop = pil_img.crop((x, y, x + crop_size[0], y + crop_size[1]))
                # score via Laplacian variance (detail measure)
                arr = np.array(crop.convert("L"), dtype=np.float32)
                lap = np.gradient(np.gradient(arr)[0])[0]
                score = float(np.var(lap))
                candidates.append(((x,y,crop_size[0],crop_size[1]), score, crop))

        candidates.sort(key=lambda x: x[1], reverse=True)
        # return top_k boxes and images
        return [(c[0], c[2]) for c in candidates[:top_k]]
