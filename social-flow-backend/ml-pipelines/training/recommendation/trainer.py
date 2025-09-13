# Orchestration of training loops
"""
trainer.py
----------
Orchestrates training of the ranking model with PyTorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from ranker import Ranker


class InteractionDataset(Dataset):
    """PyTorch dataset for user-item interactions."""

    def __init__(self, df):
        self.users = df["user_id"].values
        self.items = df["item_id"].values
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


class Trainer:
    """Handles training/validation of the Ranker model."""

    def __init__(self, num_users, num_items, lr=1e-3, batch_size=512, epochs=10, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Ranker(num_users, num_items).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.BCELoss()
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, train_df, val_df):
        train_loader = DataLoader(InteractionDataset(train_df), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(InteractionDataset(val_df), batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for users, items, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(users, items)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for users, items, labels in loader:
                users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)
                preds = self.model(users, items)
                loss = self.loss_fn(preds, labels)
                total_loss += loss.item()
        return total_loss / len(loader)
