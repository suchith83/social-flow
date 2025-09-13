# Define moderation ML model
# ============================
# File: model.py
# ============================
import torch
import torch.nn as nn
from transformers import AutoModel

class ModerationModel(nn.Module):
    """Transformer-based classifier for content moderation."""

    def __init__(self, model_name, num_labels, dropout=0.3, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
