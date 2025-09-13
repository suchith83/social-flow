# Training loop, checkpointing, optimization
# ============================
# File: trainer.py
# ============================
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import mlflow
from typing import Dict

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_scheduler("linear", self.optimizer, num_warmup_steps=int(config.warmup_ratio * total_steps), num_training_steps=total_steps)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in loop:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs["loss"]

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs["logits"], dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        return correct / total

    def train(self):
        if self.config.use_mlflow:
            mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(self.config.mlflow_experiment).experiment_id)

        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(epoch)
            acc = self.validate()
            if self.config.use_mlflow:
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_acc", acc, step=epoch)

        if self.config.use_mlflow:
            mlflow.end_run()
