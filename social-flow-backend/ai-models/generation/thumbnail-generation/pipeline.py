"""
High-level pipeline:
 - build dataset
 - instantiate generator (+ discriminator if needed)
 - train and validate
 - save checkpoints and offer inference helper

This file wires the modules together for typical usage.
"""

from .dataset import ThumbnailDataset, build_image_list_from_folder, collate_batch
from .models import ThumbnailGenerator, PatchDiscriminator
from .trainer import Trainer
from .generator import ThumbnailService
from .config import BATCH_SIZE, DEVICE, USE_GAN
from torch.utils.data import DataLoader
import os
import logging
logger = logging.getLogger("thumbnail_generation.pipeline")

class ThumbnailPipeline:
    def __init__(self, train_image_folder: str = None, val_image_folder: str = None, crop_annotations: dict = None):
        self.train_folder = train_image_folder
        self.val_folder = val_image_folder
        self.crop_annotations = crop_annotations or {}
        self.gen = ThumbnailGenerator()
        self.disc = PatchDiscriminator() if USE_GAN else None
        self.trainer = Trainer(self.gen, discriminator=self.disc, device=DEVICE)
        self.service = ThumbnailService(self.gen, device=DEVICE)

    def build_dataloaders(self, train_batch=BATCH_SIZE, val_batch=BATCH_SIZE):
        # build lists
        train_paths = build_image_list_from_folder(self.train_folder) if self.train_folder else []
        val_paths = build_image_list_from_folder(self.val_folder) if self.val_folder else []
        train_ds = ThumbnailDataset(train_paths, crop_annotations=self.crop_annotations)
        val_ds = ThumbnailDataset(val_paths, crop_annotations=self.crop_annotations, augment=False)
        train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, collate_fn=collate_batch)
        return train_loader, val_loader

    def train(self, epochs: int = None):
        train_loader, val_loader = self.build_dataloaders()
        self.trainer.train(train_loader, val_loader=val_loader, epochs=epochs or 40)

    def generate_from_path(self, image_path: str, sizes=None):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        return self.service.generate_thumbnails(img, sizes)

    def smart_crops(self, image_path: str, **kwargs):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        return self.service.smart_crop_suggestions(img, **kwargs)
