# Evaluate moderation model performance
# ============================
# File: evaluator.py
# ============================
import torch
from sklearn.metrics import classification_report

def evaluate_model(model, loader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)["logits"]
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].cpu().tolist())
    return classification_report(y_true, y_pred, output_dict=True)
