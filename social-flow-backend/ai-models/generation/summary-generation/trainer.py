"""
Training utilities for abstractive and extractive summarizers.

Features:
- Mixed precision (optional)
- Gradient accumulation
- Scheduler with warmup
- Checkpoint save/load
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from .utils import logger, save_pickle, set_seed
from .config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, GRADIENT_ACCUMULATION_STEPS, WARMUP_STEPS, MODEL_DIR

try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    # fallback simple scheduler
    get_linear_schedule_with_warmup = None

class AbstractiveTrainer:
    """
    Trainer for TransformerSeq2Seq abstractive model.
    """

    def __init__(self, model: nn.Module, tokenizer, optimizer=None, scheduler=None, device: str = DEVICE):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.stoi.get(self.tokenizer.pad_token, 0))
        self.best_loss = float("inf")

    def train(self, train_dataset, val_dataset=None, epochs: int = EPOCHS):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=self._collate_fn) if val_dataset is not None else None

        total_steps = len(train_loader) * epochs
        if get_linear_schedule_with_warmup is not None:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
            for step, batch in enumerate(pbar):
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # teacher forcing: feed previous target tokens during training
                decoder_input = target_ids[:, :-1]
                labels = target_ids[:, 1:].contiguous()

                logits = self.model(input_ids, decoder_input)  # (batch, tgt_len, vocab)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                pbar.set_postfix({"loss": epoch_loss / (step + 1)})

            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f}")

            # validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    save_path = os.path.join(MODEL_DIR, f"abstractive_best_epoch{epoch}.pt")
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Saved best model to {save_path}")

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                decoder_input = target_ids[:, :-1]
                labels = target_ids[:, 1:].contiguous()

                logits = self.model(input_ids, decoder_input)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def _collate_fn(self, batch):
        # pad variable-length sequences
        inputs, targets = zip(*batch)
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.stoi.get(self.tokenizer.pad_token, 0))
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.tokenizer.stoi.get(self.tokenizer.pad_token, 0))
        return inputs, targets

class ExtractiveTrainer:
    """
    Trainer for extractive SentenceScorer models (supervised).
    """

    def __init__(self, model: nn.Module, optimizer=None, device: str = DEVICE):
        self.model = model.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.BCELoss()

    def train(self, dataset, epochs: int = EPOCHS):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for sent_feats, labels in tqdm(loader, desc=f"Extractive Train Epoch {epoch}"):
                sent_feats = sent_feats.to(self.device)
                labels = labels.to(self.device)
                scores = self.model(sent_feats)  # (batch, num_sentences)
                loss = self.criterion(scores, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Extractive Epoch {epoch} - Avg Loss: {epoch_loss / len(loader):.4f}")
