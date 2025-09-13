# Evaluate model (accuracy, F1, etc.)
# ============================
# File: evaluator.py
# ============================
import torch
from sklearn.metrics import classification_report

def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs["logits"], dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels)
    return classification_report(y_true, y_pred, output_dict=True)
