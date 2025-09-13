# Orchestrate personalization pipeline
# ============================
# File: pipeline.py
# ============================
from .config import Config
from .data_loader import create_dataloader
from .model import TwoTowerRecommender
from .trainer import Trainer
from .evaluator import compute_ranking_metrics
import torch, json, os

def run_pipeline():
    cfg = Config()
    train_loader = create_dataloader(cfg.data.interactions_path, cfg.data.user_col, cfg.data.item_col, cfg.data.label_col, cfg.data.batch_size, cfg.data.num_workers, shuffle=True)
    val_loader = create_dataloader(cfg.data.val_path, cfg.data.user_col, cfg.data.item_col, cfg.data.label_col, cfg.data.batch_size, cfg.data.num_workers, shuffle=False)

    model = TwoTowerRecommender(cfg.model.user_vocab_size, cfg.model.item_vocab_size, cfg.model.embedding_dim, cfg.model.hidden_units, cfg.model.dropout)
    trainer = Trainer(model, train_loader, val_loader, cfg.training)
    trainer.train()

    # Evaluate
    test_loader = create_dataloader(cfg.data.test_path, cfg.data.user_col, cfg.data.item_col, cfg.data.label_col, cfg.data.batch_size, cfg.data.num_workers, shuffle=False)
    metrics = compute_ranking_metrics(model, test_loader)
    os.makedirs("reports", exist_ok=True)
    with open("reports/personalization_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
