# Common helpers
# ============================
# File: utils.py
# ============================
import torch, random, numpy as np, os

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, path): os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(model.state_dict(), path)

def load_checkpoint(model, path): model.load_state_dict(torch.load(path)); return model
