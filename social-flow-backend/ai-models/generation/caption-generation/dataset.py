"""
Dataset and DataLoader for caption generation.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from .utils import Vocabulary


class CaptionDataset(Dataset):
    def __init__(self, df, vocab: Vocabulary, transform=None, max_len=30):
        """
        df: pandas DataFrame with columns ["image_path", "caption"]
        vocab: Vocabulary object
        transform: torchvision transforms for images
        """
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        caption = [self.vocab.stoi["<SOS>"]]
        caption += self.vocab.numericalize(row["caption"])
        caption.append(self.vocab.stoi["<EOS>"])
        caption = caption[: self.max_len]

        caption_tensor = torch.tensor(caption, dtype=torch.long)
        return image, caption_tensor
