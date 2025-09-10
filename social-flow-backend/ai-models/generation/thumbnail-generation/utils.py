"""
General utilities:
 - device selection
 - seeding
 - image IO helpers (PIL <-> torch)
 - basic metrics (PSNR, SSIM)
 - saving/loading model helpers
"""

import os
import random
import math
from typing import Tuple, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pickle

from .config import MODEL_DIR, SEED

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("thumbnail_generation")

def get_device(preferred: str = "cuda"):
    return torch.device(preferred if torch.cuda.is_available() and preferred == "cuda" else "cpu")

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pil_to_tensor(img: Image.Image) -> torch.FloatTensor:
    """Convert PIL image to float tensor in range [0,1], shape CxHxW"""
    arr = TF.to_tensor(img).float()
    return arr

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor (CxHxW) in [0,1] to PIL Image"""
    t = t.clamp(0, 1).detach().cpu()
    return TF.to_pil_image(t)

def resize_tensor(t: torch.Tensor, size: Tuple[int,int]) -> torch.Tensor:
    """Resize single image tensor (CxHxW) to given size (w,h) using bilinear"""
    _, h, w = t.shape
    new_w, new_h = size
    return F.interpolate(t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)

def save_model(obj, name: str):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved model object to {path}")
    return path

def load_model(name: str):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded model object from {path}")
    return obj

# -----------------
# Metrics
# -----------------
def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute PSNR between two tensors in [0,1]. img shape CxHxW or BxCxHxW"""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((data_range ** 2) / mse)

def ssim_single(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simple SSIM wrapper using skimage if available, otherwise fall back to naive method.
    Accepts uint8 or float arrays in range [0,1].
    """
    try:
        from skimage.metrics import structural_similarity as ssim_func
        # convert to HxW or HxWxC depending on channels
        if img1.ndim == 3:
            multichannel = True
        else:
            multichannel = False
        return float(ssim_func(img1, img2, data_range=img2.max() - img2.min() if img2.max() != img2.min() else 1.0, multichannel=multichannel))
    except Exception:
        # fallback: use simple correlation measure
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        c = np.cov(img1.flatten(), img2.flatten())[0,1]
        v1 = np.var(img1.flatten())
        v2 = np.var(img2.flatten())
        denom = (v1 + v2 + 1e-8)
        return float((2 * c + 1e-8) / denom)

def compute_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    pred/target: CxHxW or BxCxHxW in [0,1] torch tensors.
    Returns PSNR and SSIM (if skimage available)
    """
    if pred.dim() == 4:
        # batch
        res = {"psnr": [], "ssim": []}
        for p, t in zip(pred, target):
            res["psnr"].append(psnr(p, t))
            p_np = (p.permute(1,2,0).cpu().numpy())
            t_np = (t.permute(1,2,0).cpu().numpy())
            res["ssim"].append(ssim_single(p_np, t_np))
        res["psnr"] = float(np.mean(res["psnr"]))
        res["ssim"] = float(np.mean(res["ssim"]))
        return res
    else:
        p = pred
        t = target
        return {"psnr": psnr(p, t), "ssim": ssim_single(p.permute(1,2,0).cpu().numpy(), t.permute(1,2,0).cpu().numpy())}
