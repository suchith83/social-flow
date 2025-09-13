# Configuration for moderation settings
# ============================
# File: config.py
# ============================
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    train_path: str = "data/moderation_train.jsonl"
    val_path: str = "data/moderation_val.jsonl"
    test_path: str = "data/moderation_test.jsonl"
    batch_size: int = 32
    num_workers: int = 4
    text_column: str = "text"
    label_column: str = "label"
    classes: List[str] = field(default_factory=lambda: ["safe", "spam", "hate", "nsfw", "harassment"])

@dataclass
class ModelConfig:
    model_name: str = "roberta-base"
    num_labels: int = 5
    dropout: float = 0.3
    freeze_encoder: bool = False

@dataclass
class TrainingConfig:
    epochs: int = 6
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation: int = 2
    mixed_precision: bool = True
    save_path: str = "checkpoints/moderation/"
    log_interval: int = 100
    use_mlflow: bool = True
    mlflow_experiment: str = "content-moderation"

@dataclass
class RuleConfig:
    profanity_list: str = "resources/profanity.txt"
    spam_threshold: float = 0.8
    nsfw_keywords: List[str] = field(default_factory=lambda: ["porn", "xxx", "nude"])

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
