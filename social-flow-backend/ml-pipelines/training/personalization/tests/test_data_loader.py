# Unit tests for data_loader
# ============================
# File: tests/test_data_loader.py
# ============================
from ml_pipelines.training.personalization.data_loader import create_dataloader
import json

def test_interaction_dataset(tmp_path):
    sample_file = tmp_path / "interactions.jsonl"
    with open(sample_file, "w") as f:
        f.write(json.dumps({"user_id": 1, "item_id": 2, "clicked": 1}) + "\n")
        f.write(json.dumps({"user_id": 3, "item_id": 4, "clicked": 0}) + "\n")

    loader = create_dataloader(str(sample_file), "user_id", "item_id", "clicked", batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    assert "user" in batch and "item" in batch and "label" in batch
