# Training loop for personalization models
# ============================
# File: trainer.py
# ============================
import torch
from torch.optim import Adam
from tqdm import tqdm
import mlflow

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in loop:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch["user"], batch["item"], batch["label"])
            loss = out["loss"]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(batch["user"], batch["item"])["logits"]
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == batch["label"].long()).sum().item()
                total += len(batch["label"])
        return correct / total

    def train(self):
        if self.config.use_mlflow:
            mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(self.config.mlflow_experiment).experiment_id)
        for epoch in range(1, self.config.epochs+1):
            loss = self.train_epoch(epoch)
            acc = self.validate()
            if self.config.use_mlflow:
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_acc", acc, step=epoch)
        if self.config.use_mlflow: mlflow.end_run()
