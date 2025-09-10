"""
Training loop for deep learning recommender models.
"""

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from .utils import logger, get_device
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS


class Trainer:
    def __init__(self, model, loss_type="bce", lr=LEARNING_RATE, device="cuda"):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.loss_fn = BCEWithLogitsLoss() if loss_type == "bce" else MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self, train_dataset, val_dataset=None, epochs=EPOCHS):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) if val_dataset else None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for user_ids, item_ids, ratings in train_loader:
                user_ids, item_ids, ratings = user_ids.to(self.device), item_ids.to(self.device), ratings.to(self.device)

                preds = self.model(user_ids, item_ids)
                loss = self.loss_fn(preds, ratings)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if val_loader:
                self.evaluate(val_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                user_ids, item_ids, ratings = user_ids.to(self.device), item_ids.to(self.device), ratings.to(self.device)
                preds = self.model(user_ids, item_ids)
                loss = self.loss_fn(preds, ratings)
                total_loss += loss.item()
        avg_val_loss = total_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
