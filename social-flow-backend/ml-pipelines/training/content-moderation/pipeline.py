# Orchestrates moderation pipeline
# ============================
# File: pipeline.py
# ============================
from transformers import AutoTokenizer
from .config import Config
from .data_loader import create_dataloader
from .model import ModerationModel
from .trainer import Trainer
from .evaluator import evaluate_model
from .rules_engine import ModerationRules
from .policy import ModerationPolicy
import torch, os, json

def run_pipeline():
    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    train_loader = create_dataloader(cfg.data.train_path, tokenizer, cfg.data.text_column, cfg.data.label_column, cfg.data.batch_size, cfg.data.num_workers, shuffle=True)
    val_loader = create_dataloader(cfg.data.val_path, tokenizer, cfg.data.text_column, cfg.data.label_column, cfg.data.batch_size, cfg.data.num_workers, shuffle=False)

    model = ModerationModel(cfg.model.model_name, cfg.model.num_labels, cfg.model.dropout, cfg.model.freeze_encoder)
    trainer = Trainer(model, train_loader, val_loader, cfg.training)
    trainer.train()

    # Evaluate
    test_loader = create_dataloader(cfg.data.test_path, tokenizer, cfg.data.text_column, cfg.data.label_column, cfg.data.batch_size, cfg.data.num_workers, shuffle=False)
    metrics = evaluate_model(model, test_loader)

    # Save metrics
    os.makedirs("reports", exist_ok=True)
    with open("reports/moderation_metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    # Initialize rules + policy for deployment
    rules = ModerationRules(cfg.rules.profanity_list, cfg.rules.nsfw_keywords, cfg.rules.spam_threshold)
    policy = ModerationPolicy(cfg.data.classes)
    return model, rules, policy
