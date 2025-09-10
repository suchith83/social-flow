"""
Training orchestration:
- Cross-validation
- Early stopping / model selection
- Save best model
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from .utils import logger, save_pickle, set_seed
from .models import GBMWrapper, TorchWrapper, LGB_AVAILABLE, TORCH_AVAILABLE
from .config import TEST_SPLIT, CV_FOLDS

class ViralTrainer:
    def __init__(self, model_type="gbm", seed=None):
        set_seed(seed)
        self.model_type = model_type
        self.model = None

    def _init_model(self, input_dim=None):
        if self.model_type == "gbm":
            self.model = GBMWrapper(task="classification")
        elif self.model_type == "nn":
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available for nn model_type")
            self.model = TorchWrapper(input_dim)
        else:
            raise ValueError("Unknown model type")

    def train(self, X: np.ndarray, y: np.ndarray, val_split=TEST_SPLIT):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, stratify=y, random_state=42)
        self._init_model(input_dim=X.shape[1] if X is not None else None)
        logger.info(f"Training {self.model_type} on {X_train.shape[0]} samples, validating on {X_val.shape[0]}")

        if self.model_type == "gbm":
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
            preds = self.model.predict_proba(X_val)
        else:
            self.model.fit(X_train, y_train, X_val, y_val)
            preds = self.model.predict_proba(X_val)

        auc = roc_auc_score(y_val, preds)
        logger.info(f"Validation AUC: {auc:.4f}")
        save_pickle(self.model, f"{self.model_type}_model.pkl")
        return self.model, auc

    def cross_validate(self, X, y, folds=CV_FOLDS):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold+1}/{folds}")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            self._init_model(input_dim=X.shape[1])
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val) if hasattr(self.model, "fit") else None)
            preds = self.model.predict_proba(X_val)
            auc = roc_auc_score(y_val, preds)
            aucs.append(auc)
            logger.info(f"Fold {fold+1} AUC: {auc:.4f}")
        logger.info(f"Mean CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        return aucs
