"""
End-to-end pipeline for caption generation.
"""

import pandas as pd
from .utils import Vocabulary
from .models import EncoderCNN, DecoderTransformer
from .trainer import CaptionTrainer
from .captioner import CaptionGenerator
from .config import DEVICE


class CaptionPipeline:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.encoder = EncoderCNN()
        self.decoder = DecoderTransformer(len(vocab.stoi))
        self.trainer = CaptionTrainer(self.encoder, self.decoder)
        self.generator = CaptionGenerator(self.encoder, self.decoder, vocab)

    def train(self, dataset):
        self.trainer.train(dataset)

    def caption_image(self, image, method="beam"):
        return self.generator.generate(image.to(DEVICE), method=method)
