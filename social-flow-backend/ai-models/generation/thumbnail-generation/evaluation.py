"""
Advanced evaluation helpers: PSNR, SSIM wrappers, LPIPS (if available), and human-friendly metrics.

LPIPS: Learned Perceptual Image Patch Similarity. If installed (lpips package), use for perceptual distance.
"""

import torch
import numpy as np
from .utils import psnr, ssim_single
import logging
logger = logging.getLogger("thumbnail_generation.evaluation")

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    pred/target: BxCxHxW or CxHxW -> returns dict of psnr, ssim, lpips (if possible)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    out = {}
    out["psnr"] = psnr(pred, target)
    # ssim average
    ssims = []
    for p,t in zip(pred, target):
        ssims.append(ssim_single((p.permute(1,2,0).cpu().numpy()), (t.permute(1,2,0).cpu().numpy())))
    out["ssim"] = float(np.mean(ssims))

    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg').to(pred.device)
        # lpips expects [-1,1]
        p_norm = (pred * 2.0) - 1.0
        t_norm = (target * 2.0) - 1.0
        lpips_val = loss_fn(p_norm, t_norm)
        out["lpips"] = float(lpips_val.mean().cpu().numpy())
    except Exception:
        out["lpips"] = None
    return out
