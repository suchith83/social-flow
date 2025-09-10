"""
Model definitions:
- Gradient Boosting Machine (LightGBM or XGBoost)
- Simple feed-forward PyTorch neural net for classification/regression
- Utility wrappers for sklearn compatibility
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import GBM_PARAMS, NN_PARAMS
from .utils import logger

# LightGBM fallback safety: try to import lightgbm, else sklearn's RandomForest
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier as RFClassifier
    from sklearn.ensemble import RandomForestRegressor as RFRegressor

# PyTorch model
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — neural models will not be usable")

@dataclass
class GBMWrapper:
    params: dict = GBM_PARAMS
    task: str = "classification"  # or 'regression'
    model: object = None

    def fit(self, X, y, eval_set=None):
        if LGB_AVAILABLE:
            lgb_train = lgb.Dataset(X, label=y)
            valid_sets = [lgb_train]
            valid_names = ["train"]
            if eval_set is not None:
                X_val, y_val = eval_set
                lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
                valid_sets.append(lgb_val); valid_names.append("val")
            self.model = lgb.train(self.params, lgb_train, valid_sets=valid_sets,
                                   valid_names=valid_names, verbose_eval=50)
        else:
            # fallback
            if self.task == "classification":
                self.model = RFClassifier(n_estimators=200, n_jobs=-1, random_state=self.params.get("random_state"))
            else:
                self.model = RFRegressor(n_estimators=200, n_jobs=-1, random_state=self.params.get("random_state"))
            self.model.fit(X, y)

    def predict_proba(self, X):
        if LGB_AVAILABLE:
            return self.model.predict(X)
        else:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)[:, 1]
            else:
                # fallback to predict
                return self.model.predict(X)

    def predict(self, X):
        if LGB_AVAILABLE:
            preds = self.model.predict(X)
            if self.task == "classification":
                return (preds >= 0.5).astype(int)
            return preds
        else:
            return self.model.predict(X)

if TORCH_AVAILABLE:
    import torch.nn.functional as F
    import torch.optim as optim
    class SimpleFeedForward(nn.Module):
        def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
            super().__init__()
            hidden_dims = hidden_dims or NN_PARAMS["hidden_dims"]
            layers = []
            in_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return torch.sigmoid(self.net(x)).squeeze(-1)

    class TorchWrapper:
        def __init__(self, input_dim, params=NN_PARAMS, device=None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SimpleFeedForward(input_dim, hidden_dims=params["hidden_dims"], dropout=params["dropout"]).to(self.device)
            self.lr = params["lr"]
            self.epochs = params["epochs"]
            self.batch_size = params["batch_size"]
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.criterion = nn.BCELoss()

        def fit(self, X, y, X_val=None, y_val=None):
            X_t = torch.tensor(X.astype(np.float32))
            y_t = torch.tensor(y.astype(np.float32))
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device); yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(loader.dataset)
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}/{self.epochs} - loss: {epoch_loss:.4f}")
            return self

        def predict_proba(self, X):
            self.model.eval()
            X_t = torch.tensor(X.astype(np.float32)).to(self.device)
            with torch.no_grad():
                return self.model(X_t).cpu().numpy()

        def predict(self, X):
            return (self.predict_proba(X) >= 0.5).astype(int)
