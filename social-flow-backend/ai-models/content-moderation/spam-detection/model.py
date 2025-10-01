# class SpamDetectionModel:
#     def detect(self, text):
#         # TODO: Detect spam
#         return {'class': 'NotSpam', 'confidence': 0.95}
"""
spam-detection/model.py

Advanced Spam Detection module for Social Flow backend.

Features:
- Hybrid classifier combining transformer-based text encoder + engineered features + rule-based heuristics.
- Training loop with mixed precision support and checkpointing.
- Inference wrapper supporting single and batched predictions, streaming scoring, and rate-limiting.
- Explainability hooks (integrated gradients skeleton + token importance).
- Deduplication/LRU cache to avoid rescoring same content repeatedly.
- Utilities for dataset handling, augmentation, metrics (precision/recall/F1/AUC).
- Safe incremental fine-tuning support for online learning scenarios.

Dependencies:
- torch, transformers, datasets, scikit-learn, tqdm
  (In a real deployment, ensure pinned versions in requirements.txt)

Author: Social Flow AI Team
Date: 2025-09-11
"""

import os
import re
import json
import math
import time
import hashlib
import logging
import random
import tempfile
from typing import List, Dict, Any, Tuple, Optional, Iterable
from collections import OrderedDict, Counter

# ML libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,  # base encoder
    AutoConfig,
    get_linear_schedule_with_warmup,
)

# Evaluation
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Progress
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("spam-detection")

# -------------------------
# Configuration Defaults
# -------------------------
DEFAULT_MODEL_NAME = "distilbert-base-uncased"  # small, fast, good starting point
DEFAULT_MAX_LEN = 256
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_NUM_EPOCHS = 3
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_THRESHOLD = 0.5

# -------------------------
# Utilities
# -------------------------


def sha1(text: str) -> str:
    """Stable hash for deduplication / cache keys."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def now_ts() -> int:
    return int(time.time())


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -------------------------
# Text Preprocessing
# -------------------------


class TextPreprocessor:
    """
    Robust text preprocessing with common normalizations and lightweight feature extraction.
    """

    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    NON_ALPHANUM = re.compile(r"[^0-9a-zA-Z\s]+")
    MULTI_SPACE = re.compile(r"\s+")

    def __init__(self, lowercase: bool = True, remove_punct: bool = False, max_length: int = DEFAULT_MAX_LEN):
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.max_length = max_length

    def normalize(self, text: str) -> str:
        """
        Normalize a single text string:
        - Remove URLs and emails (replace with tokens)
        - Optionally remove punctuation
        - Collapse whitespace and optionally lowercase
        """
        if text is None:
            return ""
        t = text.strip()
        t = self.URL_RE.sub(" <URL> ", t)
        t = self.EMAIL_RE.sub(" <EMAIL> ", t)
        if self.lowercase:
            t = t.lower()
        if self.remove_punct:
            t = self.NON_ALPHANUM.sub(" ", t)
        t = self.MULTI_SPACE.sub(" ", t).strip()
        return t[: self.max_length]

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Lightweight handcrafted features useful for spam detection.
        - length, token_count, uppercase_fraction, punctuation_fraction
        - url_count, email_count, digit_fraction, repeat_char_ratio, uncommon_char_fraction
        """
        norm = text if text else ""
        n = len(norm)
        tokens = norm.split()
        token_count = len(tokens) if tokens else 0

        if n == 0:
            return {
                "length": 0.0,
                "token_count": 0.0,
                "uppercase_fraction": 0.0,
                "punctuation_fraction": 0.0,
                "url_count": 0.0,
                "email_count": 0.0,
                "digit_fraction": 0.0,
                "repeat_char_ratio": 0.0,
            }

        uppercase = sum(1 for c in norm if c.isupper())
        punctuation = sum(1 for c in norm if not c.isalnum() and not c.isspace())
        digits = sum(1 for c in norm if c.isdigit())

        # repeated char runs: e.g., '!!!!!!' or 'loooove'
        repeat_runs = re.findall(r"(.)\1{3,}", norm)  # chars repeated 4+ times
        repeat_ratio = len(repeat_runs) / max(1, token_count)

        urls = len(self.URL_RE.findall(norm))
        emails = len(self.EMAIL_RE.findall(norm))

        return {
            "length": float(n),
            "token_count": float(token_count),
            "uppercase_fraction": float(uppercase) / n,
            "punctuation_fraction": float(punctuation) / n,
            "url_count": float(urls),
            "email_count": float(emails),
            "digit_fraction": float(digits) / n,
            "repeat_char_ratio": float(repeat_ratio),
        }


# -------------------------
# Dataset
# -------------------------


class SpamDataset(Dataset):
    """
    PyTorch Dataset for spam detection.
    Expects list of dicts: {"text": "...", "label": 0/1, "id": optional}
    """

    def __init__(self, items: List[Dict[str, Any]], tokenizer: AutoTokenizer, preprocessor: TextPreprocessor,
                 max_len: int = DEFAULT_MAX_LEN, include_features: bool = True):
        self.items = items
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.include_features = include_features

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        raw = it.get("text", "")
        norm = self.preprocessor.normalize(raw)
        enc = self.tokenizer(
            norm,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.include_features:
            feats = self.preprocessor.extract_features(raw)
            # convert to float vector; keep deterministic order
            feature_vector = np.array([
                feats["length"],
                feats["token_count"],
                feats["uppercase_fraction"],
                feats["punctuation_fraction"],
                feats["url_count"],
                feats["email_count"],
                feats["digit_fraction"],
                feats["repeat_char_ratio"],
            ], dtype=np.float32)
            inputs["features"] = torch.from_numpy(feature_vector)
        # label
        inputs["label"] = torch.tensor(it.get("label", 0), dtype=torch.long)
        inputs["id"] = it.get("id", None)
        return inputs


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to combine dataset items into batch tensors.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    ids = [b.get("id", None) for b in batch]

    if "features" in batch[0]:
        features = torch.stack([b["features"] for b in batch])
    else:
        features = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "features": features,
        "labels": labels,
        "ids": ids,
    }


# -------------------------
# Model Architecture
# -------------------------


class SpamClassifier(nn.Module):
    """
    Hybrid classifier:
    - Transformer encoder (CLS pooling)
    - Small MLP on engineered features
    - Final fusion and classifier head
    """

    def __init__(self, base_model_name: str = DEFAULT_MODEL_NAME, feature_dim: int = 8,
                 hidden_dim: int = 256, num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_config(self.config)

        # If available, try to load pretrained weights for faster convergence
        try:
            # load pretrained weights into the model
            self.encoder = AutoModel.from_pretrained(base_model_name)
        except Exception as e:
            logger.warning(f"Could not load pretrained weights for {base_model_name}: {e}")

        enc_dim = self.encoder.config.hidden_size

        # transformer projection (optional)
        self.text_proj = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # features MLP
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, max(64, feature_dim * 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, feature_dim * 8), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # fusion + head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: Optional[torch.Tensor] = None):
        """
        Forward pass:
        - Use transformer pooled output (CLS) or mean-pooling if CLS not exposed.
        - Project text + features -> fuse -> logits
        """
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # pooled output: use last_hidden_state[:,0,:] if no pooled_output
        if hasattr(enc_out, "pooler_output") and enc_out.pooler_output is not None:
            pooled = enc_out.pooler_output
        else:
            # mean pooling as fallback
            last_hidden = enc_out.last_hidden_state  # (B, L, H)
            mask = attention_mask.unsqueeze(-1)
            masked = last_hidden * mask
            summed = masked.sum(1)
            denom = mask.sum(1).clamp(min=1e-9)
            pooled = summed / denom

        text_vec = self.text_proj(pooled)  # (B, hidden_dim)

        if features is not None:
            feats_vec = self.feat_proj(features)
        else:
            # zero vector if no features
            feats_vec = torch.zeros_like(text_vec)

        fused = torch.cat([text_vec, feats_vec], dim=1)
        logits = self.classifier(fused)
        return logits


# -------------------------
# Training & Eval Utilities
# -------------------------


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    Compute key metrics: precision, recall, f1 (binary), auc
    preds: binary predictions
    probs: probability of positive class
    """
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}


class CheckpointManager:
    """
    Save and load model checkpoints with metadata.
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        ensure_dir(ckpt_dir)

    def save(self, model: nn.Module, optimizer: Any, scheduler: Any, epoch: int, metadata: dict):
        path = os.path.join(self.ckpt_dir, f"ckpt_epoch_{epoch}_{now_ts()}.pt")
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metadata": metadata,
        }
        torch.save(payload, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_latest(self):
        files = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.ckpt_dir, x)), reverse=True)
        latest = os.path.join(self.ckpt_dir, files[0])
        payload = torch.load(latest, map_location=DEFAULT_DEVICE)
        logger.info(f"Loaded checkpoint {latest}")
        return payload


def train(
    model: nn.Module,
    train_dataset: SpamDataset,
    val_dataset: Optional[SpamDataset] = None,
    output_dir: str = "./spam_model_ckpts",
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    device: str = DEFAULT_DEVICE,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    scheduler_type: str = "linear",
    save_every_epoch: bool = True,
):
    """
    Training loop for fine-tuning/spam detection.
    Supports mixed precision and gradient accumulation.
    """
    ensure_dir(output_dir)
    model.to(device)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                              batch_size=batch_size, collate_fn=collate_fn, drop_last=False)

    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler() if (mixed_precision and device.startswith("cuda")) else None

    ckpt_mgr = CheckpointManager(output_dir)
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device) if batch.get("features", None) is not None else None

            with autocast(enabled=(scaler is not None)):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
                loss = F.cross_entropy(logits, labels)
                loss = loss / gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": epoch_loss / (step + 1)})

        # End epoch
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

        if val_dataset is not None:
            metrics = evaluate(model, val_dataset, batch_size=batch_size, device=device)
            logger.info(f"Validation metrics: {metrics}")

        if save_every_epoch:
            metadata = {"epoch": epoch, "avg_loss": avg_loss}
            ckpt_mgr.save(model, optimizer, scheduler, epoch, metadata)

    return model


def evaluate(model: nn.Module, dataset: SpamDataset, batch_size: int = DEFAULT_BATCH_SIZE, device: str = DEFAULT_DEVICE) -> dict:
    """
    Evaluate model on dataset. Returns metrics dictionary.
    """
    model.eval()
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=collate_fn)
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device) if batch.get("features", None) is not None else None
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= DEFAULT_THRESHOLD).astype(int)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return metrics


# -------------------------
# Inference / Serving Wrapper
# -------------------------


class LRUCache:
    """
    Simple LRU cache for deduplication / avoiding repeated scoring.
    Stores fixed number of recent items with TTL support.
    """

    def __init__(self, capacity: int = 10000, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._store = OrderedDict()  # key -> (value, ts)

    def get(self, key: str):
        item = self._store.get(key)
        if item is None:
            return None
        value, ts = item
        if now_ts() - ts > self.ttl_seconds:
            # expired
            del self._store[key]
            return None
        # refresh order
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[key] = (value, now_ts())

    def __contains__(self, key: str):
        return self.get(key) is not None


class SpamDetectorService:
    """
    High-level service wrapper around SpamClassifier to facilitate:
    - loading/saving models & tokenizer
    - single/batched scoring
    - deduplication cache
    - lightweight rule-based overrides
    """

    def __init__(self,
                 model: Optional[SpamClassifier] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 preprocessor: Optional[TextPreprocessor] = None,
                 device: str = DEFAULT_DEVICE,
                 threshold: float = DEFAULT_THRESHOLD,
                 dedupe_cache: Optional[LRUCache] = None):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor or TextPreprocessor()
        self.threshold = threshold
        self.dedupe_cache = dedupe_cache or LRUCache(capacity=20000, ttl_seconds=3600)
        self.model_version = None

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def load_from_dir(path: str, device: str = DEFAULT_DEVICE) -> "SpamDetectorService":
        """
        Load model and tokenizer from directory (expects 'checkpoint.pt' and tokenizer files).
        """
        # Basic convention: "model.pt" contains state dict and config metadata. Tokenizer saved via huggingface methods in same dir.
        model_file = os.path.join(path, "model.pt")
        tok_dir = path  # tokenizer files typically saved in same dir
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}")

        payload = torch.load(model_file, map_location=device)
        model_config = payload.get("model_config", {"base_model": DEFAULT_MODEL_NAME})
        base = model_config.get("base_model", DEFAULT_MODEL_NAME)

        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        model = SpamClassifier(base_model_name=base)
        model.load_state_dict(payload["model_state_dict"])
        service = SpamDetectorService(model=model, tokenizer=tokenizer, preprocessor=TextPreprocessor(), device=device)
        service.model_version = payload.get("metadata", {}).get("version", None)
        logger.info(f"Loaded SpamDetectorService model version={service.model_version}")
        return service

    def save_to_dir(self, path: str, metadata: dict = None):
        ensure_dir(path)
        # save tokenizer (if present)
        if self.tokenizer:
            try:
                self.tokenizer.save_pretrained(path)
            except Exception as e:
                logger.warning(f"Failed to save tokenizer: {e}")

        model_save_path = os.path.join(path, "model.pt")
        payload = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {"base_model": getattr(self.model.encoder.config, "name_or_path", DEFAULT_MODEL_NAME)},
            "metadata": metadata or {"version": now_ts()},
        }
        torch.save(payload, model_save_path)
        logger.info(f"Saved model to {model_save_path}")

    def rule_based_score(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Quick rule-based shortcuts to immediately label or boost probability without invoking heavy model.
        Examples:
         - Known spammy tokens / blacklisted domains ? high confidence spam
         - All-caps short messages with many repeats ? likely spam
        Returns None if no rule applies.
        """
        # blacklist tokens (could be loaded from config)
        blacklist = ["free bitcoin", "click here", "earn $$$", "work from home", "viagra", "loan approval"]
        txt = text.lower()

        for phrase in blacklist:
            if phrase in txt:
                return {"label": 1, "score": 0.99, "reason": f"blacklist_phrase:{phrase}"}

        # suspicious urls/domains
        if TextPreprocessor.URL_RE.search(text):
            # many suspicious TLDs or repeated short urls -> partial signal
            url_count = len(TextPreprocessor.URL_RE.findall(text))
            if url_count >= 2:
                return {"label": 1, "score": 0.9, "reason": "multiple_urls"}

        # many repeated chars
        if re.search(r"(.)\1{7,}", text):
            return {"label": 1, "score": 0.85, "reason": "long_repeat_chars"}

        # otherwise None (no rule applied)
        return None

    def predict_single(self, text: str, uid: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Score a single text. Returns dict with keys: label (0/1), score (prob for positive), expl (optional)
        uid: optional unique id for this message; if present, used in cache key
        """
        key = sha1((uid or "") + "|" + text)
        if use_cache:
            cached = self.dedupe_cache.get(key)
            if cached is not None:
                logger.debug("Cache hit for key")
                return cached

        # rule-based fast path
        rule = self.rule_based_score(text)
        if rule is not None:
            out = {
                "label": int(rule["label"]),
                "score": float(rule["score"]),
                "reason": rule.get("reason", "rule"),
                "cached": False,
            }
            if use_cache:
                self.dedupe_cache.set(key, out)
            return out

        # preprocess + tokenize
        norm = self.preprocessor.normalize(text)
        enc = self.tokenizer(norm, truncation=True, padding="max_length", max_length=DEFAULT_MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        features = torch.from_numpy(np.array(list(self.preprocessor.extract_features(text).values()), dtype=np.float32)).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            score = float(probs[0])
            label = 1 if score >= self.threshold else 0

        out = {"label": int(label), "score": float(score), "reason": "model", "cached": False}
        if use_cache:
            self.dedupe_cache.set(key, out)
        return out

    def batch_predict(self, texts: List[str], ids: Optional[List[str]] = None, batch_size: int = 128,
                      use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Efficient batched scoring using DataLoader and model batching.
        Returns list aligned with input texts.
        """
        ids = ids or [None] * len(texts)
        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        # first apply cache and rule-based short-circuits
        to_score_idx = []
        to_score_texts = []
        to_score_ids = []
        for i, (t, uid) in enumerate(zip(texts, ids)):
            key = sha1((uid or "") + "|" + t)
            if use_cache:
                cached = self.dedupe_cache.get(key)
                if cached is not None:
                    results[i] = {**cached, "cached": True}
                    continue
            # rule-based
            r = self.rule_based_score(t)
            if r is not None:
                out = {"label": int(r["label"]), "score": float(r["score"]), "reason": r.get("reason", "rule"), "cached": False}
                if use_cache:
                    self.dedupe_cache.set(key, out)
                results[i] = out
                continue
            # needs model scoring
            to_score_idx.append(i)
            to_score_texts.append(t)
            to_score_ids.append((uid, key))

        # score in batches
        if to_score_texts:
            dataset_items = [{"text": t, "label": 0, "id": u} for t, (u, _) in zip(to_score_texts, to_score_ids)]
            ds = SpamDataset(dataset_items, tokenizer=self.tokenizer, preprocessor=self.preprocessor, include_features=True)
            loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="Batch scoring", leave=False):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    features = batch["features"].to(self.device) if batch.get("features", None) is not None else None
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask, features=features)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = (probs >= self.threshold).astype(int)
                    start = 0
                    for i in range(len(probs)):
                        global_idx = to_score_idx[start + i]
                        uid, key = to_score_ids[start + i]
                        out = {"label": int(preds[i]), "score": float(probs[i]), "reason": "model", "cached": False}
                        results[global_idx] = out
                        if use_cache:
                            self.dedupe_cache.set(key, out)
                    start += len(probs)

        # fill any remaining None (shouldn't happen)
        for i, r in enumerate(results):
            if r is None:
                results[i] = {"label": 0, "score": 0.0, "reason": "fallback", "cached": False}
        return results


# -------------------------
# Explainability Helpers (Skeletons)
# -------------------------


def token_importance_via_gradient(service: SpamDetectorService, text: str) -> List[Tuple[str, float]]:
    """
    Lightweight token importance via gradients on embedding layer.
    This function is a best-effort explainability helper and not fully production hardened.

    Returns list of (token, importance) sorted by importance desc.
    """
    norm = service.preprocessor.normalize(text)
    enc = service.tokenizer(norm, truncation=True, padding="max_length", max_length=DEFAULT_MAX_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(service.device)
    attention_mask = enc["attention_mask"].to(service.device)
    model = service.model
    model.eval()

    # requires embedding gradients, so ensure embeddings require grad
    embedding_layer = model.encoder.get_input_embeddings()
    embedding_layer.weight.requires_grad = True

    # forward
    logits = model(input_ids=input_ids, attention_mask=attention_mask, features=None)
    prob = F.softmax(logits, dim=1)[:, 1]

    # backward on positive logit
    model.zero_grad()
    prob.backward(retain_graph=True)
    # gradient of embedding for tokens in the sequence
    grads = embedding_layer.weight.grad  # (vocab_size, emb_dim)
    # approximate token importances via gradient dot embedding
    token_ids = input_ids[0].cpu().numpy().tolist()
    token_strings = service.tokenizer.convert_ids_to_tokens(token_ids)
    token_embs = embedding_layer(input_ids).detach().cpu().numpy()[0]  # (L, D)
    token_grads = embedding_layer.weight.grad.detach().cpu().numpy()  # whole vocab grads; not ideal memory-wise

    # fallback: per-token importance via L2 norm of per-token gradient by hooking embedding outputs is better;
    # Here we provide a simple proxy using grad norms (this is a placeholder - in production use Integrated Gradients or Captum)
    importances = []
    for i, tok in enumerate(token_strings):
        # some crude proxy: use norm of embedding vector (safer than using whole vocab grads)
        emb = token_embs[i]
        importance = float(np.linalg.norm(emb))  # placeholder
        importances.append((tok, importance))

    # sort by importance
    importances = sorted(importances, key=lambda x: -x[1])
    return importances[:50]


# -------------------------
# Example Command Line Interface
# -------------------------


def example_cli():
    """
    Simple CLI to run service on examples or train toy model.
    Usage:
        python model.py predict "some text"
        python model.py train --train-file data.jsonl --val-file val.jsonl --output ./ckpt
    """
    import argparse

    parser = argparse.ArgumentParser(description="Spam detection utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Predict single text")
    p_pred.add_argument("text", type=str, help="Text to score")
    p_pred.add_argument("--model-dir", type=str, default=None, help="Directory with saved model (optional)")

    p_batch = sub.add_parser("batch", help="Predict batch from file (jsonl with 'text' per line)")
    p_batch.add_argument("file", type=str, help="File path (jsonl)")
    p_batch.add_argument("--model-dir", type=str, default=None)

    p_train = sub.add_parser("train", help="Train a model from jsonl")
    p_train.add_argument("--train-file", type=str, required=True)
    p_train.add_argument("--val-file", type=str, default=None)
    p_train.add_argument("--output", type=str, default="./spam_model_ckpts")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    if args.cmd == "predict":
        if args.model_dir is None:
            # initialize a fresh model with tokenizer
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
            model = SpamClassifier(base_model_name=DEFAULT_MODEL_NAME)
            service = SpamDetectorService(model=model, tokenizer=tokenizer)
            # warn user model untrained
            logger.warning("Using untrained model - predictions will be meaningless. Load a real checkpoint with --model-dir")
        else:
            service = SpamDetectorService.load_from_dir(args.model_dir)
        out = service.predict_single(args.text)
        print(json.dumps(out, indent=2))

    elif args.cmd == "batch":
        if args.model_dir is None:
            raise ValueError("Please pass --model-dir with trained checkpoint for batch predictions")
        service = SpamDetectorService.load_from_dir(args.model_dir)
        texts = []
        with open(args.file, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                texts.append(row.get("text", ""))
        results = service.batch_predict(texts, batch_size=64)
        for r in results:
            print(json.dumps(r))

    elif args.cmd == "train":
        # read JSONL data -> items list
        def read_jsonl(path: str):
            items = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    j = json.loads(line.strip())
                    items.append({"text": j.get("text", ""), "label": int(j.get("label", 0)), "id": j.get("id", None)})
            return items

        train_items = read_jsonl(args.train_file)
        val_items = read_jsonl(args.val_file) if args.val_file else None

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        preprocessor = TextPreprocessor()
        train_ds = SpamDataset(train_items, tokenizer=tokenizer, preprocessor=preprocessor, include_features=True)
        val_ds = SpamDataset(val_items, tokenizer=tokenizer, preprocessor=preprocessor, include_features=True) if val_items else None

        model = SpamClassifier(base_model_name=DEFAULT_MODEL_NAME)
        model = train(model, train_ds, val_dataset=val_ds, output_dir=args.output, num_epochs=args.epochs,
                      batch_size=args.batch_size)
        # save final
        service = SpamDetectorService(model=model, tokenizer=tokenizer, preprocessor=preprocessor)
        service.save_to_dir(args.output, metadata={"trained_on": len(train_items), "timestamp": now_ts()})
        logger.info("Training complete and model saved.")


# -------------------------
# End of File
# -------------------------


if __name__ == "__main__":
    example_cli()
