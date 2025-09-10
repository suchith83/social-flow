"""
Evaluation for Object Recognition
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class Evaluator:
    def __init__(self, labels: list):
        self.labels = labels

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist()
        }
