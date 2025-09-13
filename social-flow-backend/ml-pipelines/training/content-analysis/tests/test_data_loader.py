# Unit tests for data_loader
# ============================
# File: tests/test_data_loader.py
# ============================
import pytest
from transformers import AutoTokenizer
from ml_pipelines.training.content_analysis.data_loader import create_dataloader

def test_dataloader_creation(tmp_path):
    sample_file = tmp_path / "sample.jsonl"
    with open(sample_file, "w") as f:
        f.write('{"text": "hello world", "label": 0}\n')
        f.write('{"text": "test text", "label": 1}\n')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader = create_dataloader(str(sample_file), tokenizer, "text", "label", batch_size=2, num_workers=0, shuffle=False)

    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch
