# Orchestrate the end-to-end training pipeline
# ============================
# File: pipeline.py
# ============================
from transformers import AutoTokenizer
from .config import Config
from .data_loader import create_dataloader
from .model import ContentAnalysisModel
from .trainer import Trainer
from .evaluator import evaluate_model
import torch
import json
import os

def run_pipeline():
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    train_loader = create_dataloader(cfg.data.train_path, tokenizer, cfg.data.text_column, cfg.data.label_column, cfg.data.batch_size, cfg.data.num_workers, shuffle=True)
    val_loader = create_dataloader(cfg.data.val_path, tokenizer, cfg.data.text_column, cfg.data.label_column, cfg.data.batch_size, cfg.data.num_workers, shuffle=False)

    model = ContentAnalysisModel(cfg.model.model_name, cfg.model.num_labels, cfg.model.dropout, cfg.model.freeze_encoder)
    trainer = Trainer(model, train_loader, val_loader, cfg.training)
    trainer.train()

    # Final evaluation
    test_loader = create_dataloader(cfg.data.test_path, tokenizer, cfg.data.text_column, cfg.data.label_column, cfg.data.batch_size, cfg.data.num_workers, shuffle=False)
    metrics = evaluate_model(model, test_loader)
    os.makedirs("reports", exist_ok=True)
    with open("reports/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
