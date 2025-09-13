# Centralized configuration for training
# ============================
# File: config.py
# ============================
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    test_path: str = "data/test.jsonl"
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    text_column: str = "text"
    label_column: str = "label"

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    num_labels: int = 5
    dropout: float = 0.3
    freeze_encoder: bool = False

@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    log_interval: int = 50
    save_path: str = "checkpoints/"
    use_mlflow: bool = True
    mlflow_experiment: str = "content-analysis-training"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
