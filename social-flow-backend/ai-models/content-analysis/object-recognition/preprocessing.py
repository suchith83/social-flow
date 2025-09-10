"""
Image Preprocessing Module

Handles:
- Loading
- Resizing
- Normalization
- Augmentation
"""

from torchvision import transforms
from PIL import Image
import torch
from .config import ObjectConfig


class ImagePreprocessor:
    def __init__(self, img_size=ObjectConfig.IMG_SIZE):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, filepath: str) -> torch.Tensor:
        """Load and preprocess image"""
        img = Image.open(filepath).convert("RGB")
        return self.transform(img).unsqueeze(0)  # add batch dim
