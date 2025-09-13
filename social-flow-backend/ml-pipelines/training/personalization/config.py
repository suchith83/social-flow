# Centralized configuration for personalization
# ============================
# File: config.py
# ============================
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    interactions_path: str = "data/interactions_train.jsonl"
    val_path: str = "data/interactions_val.jsonl"
    test_path: str = "data/interactions_test.jsonl"
    batch_size: int = 256
    num_workers: int = 4
    user_col: str = "user_id"
    item_col: str = "item_id"
    label_col: str = "clicked"

@dataclass
class ModelConfig:
    user_vocab_size: int = 50000
    item_vocab_size: int = 100000
    embedding_dim: int = 128
    hidden_units: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    log_interval: int = 100
    save_path: str = "checkpoints/personalization/"
    use_mlflow: bool = True
    mlflow_experiment: str = "personalization"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
