# Define ML model architecture
# ============================
# File: model.py
# ============================
import torch
import torch.nn as nn
from transformers import AutoModel

class ContentAnalysisModel(nn.Module):
    """Transformer-based classifier for content analysis."""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
