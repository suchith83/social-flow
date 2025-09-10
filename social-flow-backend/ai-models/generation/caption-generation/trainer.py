"""
Training loop for caption generation models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import logger
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE


class CaptionTrainer:
    def __init__(self, encoder, decoder):
        self.encoder = encoder.to(DEVICE)
        self.decoder = decoder.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        params = list(self.decoder.parameters()) + list(self.encoder.fc.parameters())
        self.optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    def train(self, dataset, epochs=EPOCHS):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self._collate_fn)

        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0
            for imgs, captions in loader:
                imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)

                features = self.encoder(imgs)
                outputs = self.decoder(features, captions[:, :-1])
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def _collate_fn(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        return images, captions
