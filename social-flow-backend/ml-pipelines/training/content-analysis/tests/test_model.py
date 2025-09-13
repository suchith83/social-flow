# Unit tests for model
# ============================
# File: tests/test_model.py
# ============================
import torch
from ml_pipelines.training.content_analysis.model import ContentAnalysisModel

def test_model_forward():
    model = ContentAnalysisModel("bert-base-uncased", num_labels=2)
    inputs = {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16),
        "labels": torch.tensor([0, 1])
    }
    outputs = model(**inputs)
    assert "logits" in outputs
    assert outputs["logits"].shape[1] == 2
