# Training loop orchestration
"""
trainer.py
----------
Handles training and validation of the ViralPredictor.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from viral_predictor import ViralPredictor
from evaluator import evaluate_classification


class ViralDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    def __init__(self, input_dim, lr=1e-3, batch_size=256, epochs=10, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ViralPredictor(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.BCELoss()
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X_train, y_train, X_val, y_val):
        train_loader = DataLoader(ViralDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(ViralDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X)
                loss = self.loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Val Metrics={val_metrics}")

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
        return evaluate_classification(np.array(all_labels), np.array(all_preds))
