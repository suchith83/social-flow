"""
Dataset utilities for training thumbnail generation.

Supports:
 - directory of source images + optional annotated crop boxes
 - video frame extraction helper (lightweight)
 - stabilized transforms, multi-aspect sampling (generate target thumbnails at various aspect ratios)
"""

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import glob
import torchvision.transforms as T
import torch
from typing import List, Tuple, Optional
from .utils import pil_to_tensor
from .config import SRC_SIZE, THUMBNAIL_SIZES

import logging
logger = logging.getLogger("thumbnail_generation.dataset")

class ThumbnailDataset(Dataset):
    """
    Directory-based dataset:
    - root_dir contains images (or frames) in nested folders OR a csv listing image paths
    - optional 'crop_annotations' is a dict mapping image_path -> list of crop boxes ((x,y,w,h)) representing good crops
      If provided, we sample either a crop (positive) or random crop (negative) depending on training mode.
    """

    def __init__(self, image_paths: List[str], crop_annotations: Optional[dict] = None,
                 src_size: Tuple[int,int] = SRC_SIZE, thumb_sizes: List[Tuple[int,int]] = THUMBNAIL_SIZES,
                 augment: bool = True):
        self.paths = image_paths
        self.crop_ann = crop_annotations or {}
        self.src_size = src_size
        self.thumb_sizes = thumb_sizes
        self.augment = augment

        # basic transforms
        self.base_transform = T.Compose([
            T.Resize((src_size[1], src_size[0])),  # PIL expects (h,w)
            T.RandomHorizontalFlip(p=0.5) if augment else T.Lambda(lambda x: x),
        ])

    def __len__(self):
        return len(self.paths)

    def _sample_crop(self, img_w, img_h, ann_list=None, crop_size=(320,180)):
        """
        Sample crop. If ann_list provided, sample around annotated box with jitter; else random crop.
        crop_size is (w,h)
        Returns box (x,y,w,h)
        """
        w, h = crop_size
        if ann_list:
            box = random.choice(ann_list)
            bx, by, bw, bh = box
            # jitter around center
            cx = int(bx + bw/2 + random.uniform(-0.15*bw, 0.15*bw))
            cy = int(by + bh/2 + random.uniform(-0.15*bh, 0.15*bh))
            x = max(0, min(img_w - w, cx - w//2))
            y = max(0, min(img_h - h, cy - h//2))
            return (x,y,w,h)
        else:
            x = random.randint(0, max(0, img_w - w))
            y = random.randint(0, max(0, img_h - h))
            return (x,y,w,h)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.base_transform(img)

        # source tensor (resized to src_size)
        src_tensor = pil_to_tensor(img)

        # choose a thumbnail target size randomly (to teach multi-aspect outputs)
        target_size = random.choice(self.thumb_sizes)
        ann = self.crop_ann.get(path, None)

        # sample crop
        img_w, img_h = img.size
        x,y,w,h = self._sample_crop(img_w, img_h, ann, crop_size=target_size)
        # crop and resize to target size exactly (for target)
        cropped = img.crop((x, y, x+w, y+h)).resize((target_size[0], target_size[1]), Image.LANCZOS)
        target_tensor = pil_to_tensor(cropped)

        return src_tensor, target_tensor, path, (x,y,w,h), target_size

def build_image_list_from_folder(root_dir: str, extensions: List[str] = ["jpg", "png", "jpeg"]) -> List[str]:
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True))
    logger.info(f"Found {len(files)} images in {root_dir}")
    return files

def collate_batch(batch):
    # pad/stack dynamic sizes: for simplicity training sampler returns pairs resized to SRC_SIZE & target size already
    srcs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    paths = [b[2] for b in batch]
    boxes = [b[3] for b in batch]
    sizes = [b[4] for b in batch]
    return srcs, targets, paths, boxes, sizes
