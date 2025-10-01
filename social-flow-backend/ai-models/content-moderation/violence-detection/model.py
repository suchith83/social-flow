# class ViolenceDetectionModel:
#     def detect(self, image):
#         # TODO: Detect violence
#         return {'class': 'Safe', 'confidence': 0.95}


"""
violence-detection/model.py

Advanced Violence Detection module for Social Flow backend.

Capabilities:
- Image-level violence classifier (transfer learning backbone).
- Frame-sequence (video) violence detector using lightweight temporal modelling.
- Fusion with optional object-detection signals (hook for integrations with YOLO/DETR).
- Explainability via Grad-CAM (image) and per-frame scoring for video.
- Training/evaluation utilities (mixed precision, checkpointing, metrics).
- Inference wrapper for single image, batched images, single video, batched videos.
- Deduplication LRU cache and rule-based quick checks.
- CLI for predict/train operations.

Design goals:
- Modular, production-friendly code that is easy to extend.
- Reasonable defaults for quick experiments; can be swapped to stronger backbones.
- Clear docstrings and comments to make the flow understandable.

Author: Social Flow AI Team
Date: 2025-09-11
"""

import os
import sys
import time
import json
import math
import logging
import hashlib
import tempfile
from typing import List, Tuple, Dict, Optional, Any
from collections import OrderedDict

# Core ML libs
import numpy as np
from PIL import Image
import cv2  # for video frame reading and simple image ops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode

# Mixed precision
from torch.cuda.amp import autocast, GradScaler

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Progress
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("violence-detection")

# -------------------------
# Defaults / Config
# -------------------------
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
FRAME_SAMPLE_RATE = 2  # sample 1 frame every N frames by default
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SEQ_LEN = 16  # max frames in sequence for temporal model
DEFAULT_THRESHOLD = 0.5
CHECKPOINT_DIR = "./violence_ckpts"

# -------------------------
# Utilities
# -------------------------


def now_ts() -> int:
    return int(time.time())


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# -------------------------
# Preprocessing
# -------------------------


def get_image_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Standardized preprocessing for image classifier.
    Using ImageNet normalization as default since we use ImageNet pretrained backbones.
    """
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# -------------------------
# Datasets
# -------------------------


class ImageViolenceDataset(Dataset):
    """
    Dataset for image-level violence classification.
    Expects items: [{"image_path": "...", "label": 0/1, "id": optional}, ...]
    """

    def __init__(self, items: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None):
        self.items = items
        self.transform = transform or get_image_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        path = rec["image_path"]
        label = int(rec.get("label", 0))
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {path}: {e}")
            # return a zero image and label 0 to avoid crashing (caller should handle)
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        img_t = self.transform(img)
        return {"image": img_t, "label": torch.tensor(label, dtype=torch.long), "id": rec.get("id", None), "path": path}


class VideoFrameSequenceDataset(Dataset):
    """
    Dataset for video sequences. Each item is a sequence of frames sampled from a video.
    Expects items: [{"video_path": "...", "label": 0/1, "start_sec": optional, "id": optional}, ...]

    Note: For simplicity, this class reads frames on the fly. In production you'd use pre-extracted frame tensors or a video decoder service.
    """

    def __init__(self, items: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None,
                 max_seq_len: int = DEFAULT_MAX_SEQ_LEN, frame_sample_rate: int = FRAME_SAMPLE_RATE):
        self.items = items
        self.transform = transform or get_image_transform()
        self.max_seq_len = max_seq_len
        self.frame_sample_rate = frame_sample_rate

    def __len__(self):
        return len(self.items)

    def _read_frames(self, video_path: str, start_sec: Optional[float] = None) -> List[np.ndarray]:
        """
        Read frames from a video file with OpenCV. Return list of PIL images.
        We sample frames every self.frame_sample_rate frames to limit sequence length.
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            # Seek to start time if provided
            if start_sec:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
            idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if idx % self.frame_sample_rate == 0:
                    # convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame_rgb)
                    frames.append(pil)
                    if len(frames) >= self.max_seq_len:
                        break
                idx += 1
            cap.release()
        except Exception as e:
            logger.error(f"Error reading frames from {video_path}: {e}")
        return frames

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        vpath = rec["video_path"]
        start_sec = rec.get("start_sec", None)
        label = int(rec.get("label", 0))
        frames = self._read_frames(vpath, start_sec=start_sec)
        if not frames:
            # fallback to zero tensors
            zero = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            seq = torch.stack([zero] * self.max_seq_len)
        else:
            proc = [self.transform(f) for f in frames]
            # pad/truncate to max_seq_len
            if len(proc) < self.max_seq_len:
                padding = [torch.zeros_like(proc[0]) for _ in range(self.max_seq_len - len(proc))]
                proc.extend(padding)
            else:
                proc = proc[:self.max_seq_len]
            seq = torch.stack(proc)  # (seq_len, C, H, W)
        return {"frames": seq, "label": torch.tensor(label, dtype=torch.long), "id": rec.get("id", None), "path": vpath}


def collate_image_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    paths = [b["path"] for b in batch]
    ids = [b.get("id", None) for b in batch]
    return {"images": images, "labels": labels, "paths": paths, "ids": ids}


def collate_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch frames shape: (B, seq_len, C, H, W)
    frames = torch.stack([b["frames"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    paths = [b["path"] for b in batch]
    ids = [b.get("id", None) for b in batch]
    return {"frames": frames, "labels": labels, "paths": paths, "ids": ids}


# -------------------------
# Model components
# -------------------------


class SpatialBackbone(nn.Module):
    """
    Image backbone for spatial feature extraction. Default: EfficientNet-B0 or ResNet50.
    We expose a pooling output for Grad-CAM use.
    """

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, output_dim: int = 512):
        super().__init__()
        name = backbone_name.lower()
        if name.startswith("resnet"):
            self.net = models.resnet50(pretrained=pretrained) if name == "resnet50" else models.resnet34(pretrained=pretrained)
            # remove final fc
            in_feat = self.net.fc.in_features
            self.net.fc = nn.Identity()
            self.project = nn.Sequential(
                nn.Linear(in_feat, output_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif name.startswith("efficientnet"):
            # use torchvision efficientnet
            try:
                self.net = models.efficientnet_b0(pretrained=pretrained)
                in_feat = self.net.classifier[1].in_features
                self.net.classifier = nn.Identity()
                self.project = nn.Sequential(
                    nn.Linear(in_feat, output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            except Exception:
                # fallback to resnet
                self.net = models.resnet50(pretrained=pretrained)
                in_feat = self.net.fc.in_features
                self.net.fc = nn.Identity()
                self.project = nn.Sequential(
                    nn.Linear(in_feat, output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
        else:
            raise ValueError("Unsupported backbone")

        # placeholder for feature maps required by Grad-CAM
        self._feature_maps = None
        # register forward hook to capture last conv feature maps for explainability
        self._register_hooks()

    def _register_hooks(self):
        """
        For ResNet and EfficientNet we capture the output of the last convolutional block.
        This code tries a few sensible attribute names.
        """
        def hook(module, input, output):
            self._feature_maps = output.detach()

        # try common names
        candidates = []
        if hasattr(self.net, "layer4"):
            candidates.append(self.net.layer4)
        if hasattr(self.net, "features"):
            # efficientnet stores features in .features
            candidates.append(self.net.features)
        if candidates:
            # attach hook to last candidate module
            try:
                candidates[-1].register_forward_hook(hook)
            except Exception:
                logger.debug("Could not register forward hook on backbone; Grad-CAM may not work.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward through base and project
        feat = self.net(x)
        out = self.project(feat)
        return out

    def get_feature_maps(self):
        return self._feature_maps


class TemporalModule(nn.Module):
    """
    Lightweight temporal module that ingests a sequence of spatial features and models short-term dynamics.
    Options: ConvLSTM, Temporal Conv, or simple GRU over pooled features.
    We'll implement a simple bidirectional GRU over per-frame features (efficient).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, seq_feats: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq_feats: (B, seq_len, feat_dim)
        return: (B, out_dim) pooled final representation
        """
        # Optionally could apply masking for padded frames; for now assume fixed length or padded with zeros
        outputs, hidden = self.gru(seq_feats)  # outputs: (B, seq_len, out_dim)
        # use mean pooling over time
        pooled = outputs.mean(dim=1)
        return pooled


class DetectorHead(nn.Module):
    """
    Final classifier that fuses spatial & temporal features + optional object cues.
    Produces:
      - logits (2 classes: non-violent, violent)
      - calibrated 'threat score' in [0,1]
      - optional auxiliary outputs (e.g., object threat cues)
    """

    def __init__(self, spatial_dim: int = 512, temporal_dim: int = 512, object_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.object_dim = object_dim

        fusion_dim = spatial_dim + temporal_dim + object_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)  # binary logits
        # small calibration head mapping logits -> threat score [0,1]
        self.calibrator = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor, object_vec: Optional[torch.Tensor] = None):
        if object_vec is None:
            # zero object embedding
            object_vec = torch.zeros(spatial_vec.size(0), self.object_dim, device=spatial_vec.device)
        fused = torch.cat([spatial_vec, temporal_vec, object_vec], dim=1)
        h = self.fusion(fused)
        logits = self.classifier(h)
        score = self.calibrator(F.softmax(logits, dim=1))
        return logits, score.squeeze(-1)  # logits: (B,2), score: (B,)


# -------------------------
# Full Model
# -------------------------


class ViolenceModel(nn.Module):
    """
    Full model that exposes both image-level classification and sequence-level classification.
    For single-image inference we only use spatial backbone + fusion head (temporal part receives zeros).
    For sequence inference, we process each frame through spatial backbone, aggregate via temporal module, and classify.
    """

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True,
                 spatial_dim: int = 512, temporal_hidden: int = 256, object_dim: int = 64):
        super().__init__()
        self.spatial = SpatialBackbone(backbone_name=backbone_name, pretrained=pretrained, output_dim=spatial_dim)
        self.temporal = TemporalModule(input_dim=spatial_dim, hidden_dim=temporal_hidden)
        # temporal out dim depends on bidirectional: TemporalModule sets out_dim attribute dynamically
        temporal_out_dim = self.temporal.out_dim
        self.head = DetectorHead(spatial_dim=spatial_dim, temporal_dim=temporal_out_dim, object_dim=object_dim)

    def forward_image(self, images: torch.Tensor):
        """
        images: (B, C, H, W)
        """
        spatial_vec = self.spatial(images)  # (B, spatial_dim)
        # temporal_vec as zeros (no temporal info)
        temporal_vec = torch.zeros(spatial_vec.size(0), self.temporal.out_dim, device=spatial_vec.device)
        logits, score = self.head(spatial_vec, temporal_vec, None)
        return logits, score

    def forward_sequence(self, frames: torch.Tensor):
        """
        frames: (B, seq_len, C, H, W)
        We'll apply spatial backbone frame-wise (efficiently by flattening batch/seq dims), then temporal module.
        """
        B, S, C, H, W = frames.shape
        flattened = frames.view(B * S, C, H, W)
        spatial_feats = self.spatial(flattened)  # (B*S, spatial_dim)
        spatial_feats = spatial_feats.view(B, S, -1)  # (B, S, spatial_dim)
        temporal_vec = self.temporal(spatial_feats)  # (B, temporal_out_dim)
        # we also produce per-frame logits optionally by applying head on each frame (for per-frame explainability)
        # per-frame spatial logits (no temporal)
        per_frame_logits = []
        # to save compute, project frames through a small classifier head if needed (placeholder)
        for s in range(S):
            f = spatial_feats[:, s, :]  # (B, spatial_dim)
            # temporal for single frame is zero
            t0 = torch.zeros(B, self.temporal.out_dim, device=f.device)
            logit, _ = self.head(f, t0, None)
            per_frame_logits.append(logit)
        per_frame_logits = torch.stack(per_frame_logits, dim=1)  # (B, S, 2)
        # final logits & score
        logits, score = self.head(spatial_feats.mean(dim=1), temporal_vec, None)
        return logits, score, per_frame_logits


# -------------------------
# Training & Evaluation Helpers
# -------------------------


class CheckpointManager:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        ensure_dir(ckpt_dir)

    def save(self, model: nn.Module, optimizer: Any, scheduler: Any, epoch: int, metadata: dict):
        path = os.path.join(self.ckpt_dir, f"violence_ckpt_epoch_{epoch}_{now_ts()}.pt")
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "metadata": metadata,
        }
        torch.save(payload, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_latest(self):
        files = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.ckpt_dir, x)), reverse=True)
        latest = os.path.join(self.ckpt_dir, files[0])
        payload = torch.load(latest, map_location=DEFAULT_DEVICE)
        logger.info(f"Loaded checkpoint: {latest}")
        return payload


def compute_metrics_binary(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel() if len(np.unique(labels)) > 1 else (0, 0, 0, 0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def train_image_model(model: ViolenceModel, train_ds: ImageViolenceDataset, val_ds: Optional[ImageViolenceDataset] = None,
                      epochs: int = 5, batch_size: int = 16, lr: float = 2e-4, weight_decay: float = 1e-4,
                      device: str = DEFAULT_DEVICE, mixed_precision: bool = True, ckpt_dir: str = CHECKPOINT_DIR):
    """
    Simple training loop for image-level model. For video training, use train_sequence_model or similar wrapper.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = math.ceil(len(train_ds) / batch_size) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    scaler = GradScaler() if (mixed_precision and device.startswith("cuda")) else None

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_image_batch, num_workers=4)
    ckpt_mgr = CheckpointManager(ckpt_dir)
    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Train Image Epoch {epoch}", leave=False)
        running_loss = 0.0
        for batch in pbar:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=(scaler is not None)):
                logits, score = model.forward_image(images)
                loss = F.cross_entropy(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
        logger.info(f"Epoch {epoch} finished. Avg Loss: {running_loss / len(loader):.4f}")
        # validation
        if val_ds:
            metrics = evaluate_image_model(model, val_ds, batch_size=batch_size, device=device)
            logger.info(f"Validation metrics: {metrics}")
        ckpt_mgr.save(model, optimizer, scheduler, epoch, metadata={"epoch_loss": running_loss / len(loader), "epoch": epoch})
    return model


def evaluate_image_model(model: ViolenceModel, dataset: ImageViolenceDataset, batch_size: int = 16, device: str = DEFAULT_DEVICE):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_image_batch, num_workers=4)
    labels = []
    preds = []
    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate Image", leave=False):
            images = batch["images"].to(device)
            y = batch["labels"].cpu().numpy()
            logits, score = model.forward_image(images)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (p >= DEFAULT_THRESHOLD).astype(int)
            labels.extend(y.tolist())
            preds.extend(pred.tolist())
            probs.extend(p.tolist())
    return compute_metrics_binary(np.array(labels), np.array(preds), np.array(probs))


def train_sequence_model(model: ViolenceModel, train_ds: VideoFrameSequenceDataset, val_ds: Optional[VideoFrameSequenceDataset] = None,
                         epochs: int = 5, batch_size: int = 4, lr: float = 2e-4, device: str = DEFAULT_DEVICE,
                         mixed_precision: bool = True, ckpt_dir: str = CHECKPOINT_DIR):
    """
    Training loop for sequence-level (video) model. Because frames are heavier, default batch size is small.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = math.ceil(len(train_ds) / batch_size) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    scaler = GradScaler() if (mixed_precision and device.startswith("cuda")) else None

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_video_batch, num_workers=4)
    ckpt_mgr = CheckpointManager(ckpt_dir)
    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Train Seq Epoch {epoch}", leave=False)
        running_loss = 0.0
        for batch in pbar:
            frames = batch["frames"].to(device)  # (B, S, C, H, W)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=(scaler is not None)):
                logits, score, _ = model.forward_sequence(frames)
                loss = F.cross_entropy(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
        logger.info(f"Epoch {epoch} finished. Avg Loss: {running_loss / len(loader):.4f}")
        if val_ds:
            metrics = evaluate_sequence_model(model, val_ds, batch_size=batch_size, device=device)
            logger.info(f"Validation metrics: {metrics}")
        ckpt_mgr.save(model, optimizer, scheduler, epoch, metadata={"epoch_loss": running_loss / len(loader), "epoch": epoch})
    return model


def evaluate_sequence_model(model: ViolenceModel, dataset: VideoFrameSequenceDataset, batch_size: int = 2, device: str = DEFAULT_DEVICE):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_video_batch, num_workers=4)
    labels = []
    preds = []
    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate Seq", leave=False):
            frames = batch["frames"].to(device)
            y = batch["labels"].cpu().numpy()
            logits, score, per_frame_logits = model.forward_sequence(frames)
            p = score.cpu().numpy()
            pred = (p >= DEFAULT_THRESHOLD).astype(int)
            labels.extend(y.tolist())
            preds.extend(pred.tolist())
            probs.extend(p.tolist())
    return compute_metrics_binary(np.array(labels), np.array(preds), np.array(probs))


# -------------------------
# Explainability (Grad-CAM)
# -------------------------


def grad_cam_image(model: ViolenceModel, image_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a single image (tensor shape: (C,H,W) or (1,C,H,W)).
    Returns heatmap as numpy array of same HxW (values 0..1).
    Notes:
      - This is a simplified implementation for back-of-the-envelope interpretability.
      - For robust explainability, consider using Captum or other libraries.
    """
    model.eval()
    device = next(model.parameters()).device
    if image_tensor.ndim == 3:
        inp = image_tensor.unsqueeze(0).to(device)
    else:
        inp = image_tensor.to(device)

    # Ensure feature maps are captured by SpatialBackbone via hook
    model.zero_grad()
    logits, score = model.forward_image(inp)
    # select target logit
    target = logits[:, class_idx].sum()
    target.backward(retain_graph=True)

    # get captured feature maps
    feature_maps = model.spatial.get_feature_maps()  # (B, C, Hf, Wf)
    if feature_maps is None:
        logger.warning("Feature maps not found. Ensure hook registration in SpatialBackbone worked.")
        # fallback to zeros
        H, W = inp.shape[-2], inp.shape[-1]
        return np.zeros((H, W), dtype=np.float32)

    grads = None
    # get gradients of the last conv layer by accessing stored grads on the tensors
    # Note: This simplistic approach may not work for all backbones; production code should use hooks to capture grads and activations.
    try:
        grads = feature_maps.grad
    except Exception:
        # try to get gradient using registered hooks (not implemented here)
        grads = None

    if grads is None:
        # compute gradients w.r.t. input as fallback and use input-gradient magnitude
        input_grad = inp.grad
        if input_grad is None:
            # no gradients available; return zero map
            H, W = inp.shape[-2], inp.shape[-1]
            return np.zeros((H, W), dtype=np.float32)
        g = input_grad.squeeze().cpu().numpy()
        g_norm = np.mean(np.abs(g), axis=0)
        g_norm = (g_norm - g_norm.min()) / (g_norm.max() - g_norm.min() + 1e-9)
        return g_norm

    # Aggregate grads to weights and compute weighted sum of feature maps
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    cam = (weights * feature_maps).sum(dim=1).squeeze(0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (inp.shape[-1], inp.shape[-2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam


# -------------------------
# Inference Service Wrapper
# -------------------------


class LRUCache:
    """
    Simple in-memory LRU cache with TTL for deduplication and quick hits.
    """

    def __init__(self, capacity: int = 10000, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._store = OrderedDict()  # key -> (value, ts)

    def get(self, key: str):
        item = self._store.get(key)
        if item is None:
            return None
        value, ts = item
        if now_ts() - ts > self.ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[key] = (value, now_ts())

    def __contains__(self, key: str):
        return self.get(key) is not None


class ViolenceDetectorService:
    """
    High-level service that encapsulates the model and provides
    image-level and video-level inference utilities, caching, and rule shortcuts.
    """

    def __init__(self, model: ViolenceModel, device: str = DEFAULT_DEVICE, threshold: float = DEFAULT_THRESHOLD,
                 cache: Optional[LRUCache] = None):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.cache = cache or LRUCache(capacity=20000)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_from_checkpoint(path: str, device: str = DEFAULT_DEVICE) -> "ViolenceDetectorService":
        payload = torch.load(path, map_location=device)
        # We assume the checkpoint payload contains model_state_dict and model_config
        model_config = payload.get("metadata", {}).get("model_config", {})
        backbone_name = model_config.get("backbone_name", "resnet50")
        model = ViolenceModel(backbone_name=backbone_name, pretrained=False)
        model.load_state_dict(payload["model_state_dict"])
        service = ViolenceDetectorService(model=model, device=device)
        logger.info(f"Loaded ViolenceDetectorService from {path}")
        return service

    def rule_quick_check(self, image: Optional[Image.Image] = None, text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fast heuristics that may short-circuit the model for obvious cases:
        - If a frame contains no people (via simple skin detection or a light object detector output), score low.
        - If metadata or attached text contains explicit violence keywords, raise quick flag.
        NOTE: These heuristics are conservative; they are intended to reduce compute for obvious non-violent content.
        """
        if text:
            # crude list of violent keywords (should be external config in prod)
            keywords = ["kill", "murder", "stab", "shoot", "blood", "fight", "assault", "beat"]
            tx = text.lower()
            for kw in keywords:
                if kw in tx:
                    return {"label": 1, "score": 0.9, "reason": f"keyword:{kw}"}
        if image:
            # quick skin detection or no-people heuristic: compute edge density; if very low, likely non-violent static image
            npimg = np.array(image.convert("L"))
            edges = cv2.Canny(npimg, 100, 200)
            edge_density = edges.sum() / (edges.size + 1e-9)
            if edge_density < 0.001:
                return {"label": 0, "score": 0.01, "reason": "low_edge_density"}
        return None

    def predict_image(self, image_path: str, use_cache: bool = True) -> Dict[str, Any]:
        key = sha1(image_path)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return {**cached, "cached": True}
        # try rule quick check
        try:
            pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            return {"error": str(e)}
        rule = self.rule_quick_check(image=pil)
        if rule:
            if use_cache:
                self.cache.set(key, rule)
            return {**rule, "cached": False}
        # preprocess
        transform = get_image_transform()
        img_t = transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, score = self.model.forward_image(img_t)
            prob = float(score.item())
            label = 1 if prob >= self.threshold else 0
            logits_np = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
        out = {"label": int(label), "score": prob, "prob_distribution": logits_np, "reason": "model", "cached": False}
        if use_cache:
            self.cache.set(key, out)
        return out

    def predict_video(self, video_path: str, seq_start_sec: Optional[float] = None, use_cache: bool = True) -> Dict[str, Any]:
        key = sha1(video_path + str(seq_start_sec))
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return {**cached, "cached": True}
        # quick heuristic could be applied (e.g., metadata)
        # assemble dataset item and run through VideoFrameSequenceDataset for single item
        ds_item = [{"video_path": video_path, "label": 0, "start_sec": seq_start_sec, "id": None}]
        vds = VideoFrameSequenceDataset(ds_item)
        batch = collate_video_batch([vds[0]])
        frames = batch["frames"].to(self.device).unsqueeze(0) if batch["frames"].ndim == 4 else batch["frames"].to(self.device)  # (1,S,C,H,W)
        # if dataset returns (S,C,H,W) convert to (1,S,C,H,W)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        with torch.no_grad():
            logits, score, per_frame_logits = self.model.forward_sequence(frames)
            prob = float(score.item())
            label = 1 if prob >= self.threshold else 0
            per_frame_probs = F.softmax(per_frame_logits, dim=2)[:, :, 1].cpu().numpy().tolist()  # (B,S)
            logits_np = F.softmax(logits, dim=1).cpu().numpy().tolist()
        out = {"label": int(label), "score": prob, "prob_distribution": logits_np[0], "per_frame_probs": per_frame_probs[0], "reason": "model", "cached": False}
        if use_cache:
            self.cache.set(key, out)
        return out


# -------------------------
# CLI
# -------------------------


def cli():
    """
    Simple CLI for quick image/video prediction and training stubs.

    Examples:
      python model.py predict-image /path/to/img.jpg
      python model.py predict-video /path/to/video.mp4
      python model.py train-image --train-json train.jsonl --val-json val.jsonl --epochs 3
      python model.py train-video --train-json train.jsonl --val-json val.jsonl --epochs 2
    """
    import argparse

    parser = argparse.ArgumentParser(description="Violence detection utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("predict-image", help="Predict single image")
    p_img.add_argument("image", type=str)

    p_vid = sub.add_parser("predict-video", help="Predict single video (sequence)")
    p_vid.add_argument("video", type=str)
    p_vid.add_argument("--start-sec", type=float, default=None)

    p_train_img = sub.add_parser("train-image")
    p_train_img.add_argument("--train-json", type=str, required=True, help="JSONL with {'image_path','label'}")
    p_train_img.add_argument("--val-json", type=str, default=None)
    p_train_img.add_argument("--epochs", type=int, default=3)

    p_train_vid = sub.add_parser("train-video")
    p_train_vid.add_argument("--train-json", type=str, required=True, help="JSONL with {'video_path','label'}")
    p_train_vid.add_argument("--val-json", type=str, default=None)
    p_train_vid.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.cmd == "predict-image":
        # init model with default weights (untrained)
        model = ViolenceModel(backbone_name="resnet50", pretrained=True)
        service = ViolenceDetectorService(model)
        out = service.predict_image(args.image)
        print(json.dumps(out, indent=2))

    elif args.cmd == "predict-video":
        model = ViolenceModel(backbone_name="resnet50", pretrained=True)
        service = ViolenceDetectorService(model)
        out = service.predict_video(args.video, seq_start_sec=args.start_sec)
        print(json.dumps(out, indent=2))

    elif args.cmd == "train-image":
        def read_jsonl(path):
            items = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    j = json.loads(line)
                    items.append({"image_path": j.get("image_path"), "label": int(j.get("label", 0)), "id": j.get("id", None)})
            return items
        train_items = read_jsonl(args.train_json)
        val_items = read_jsonl(args.val_json) if args.val_json else None
        model = ViolenceModel(backbone_name="resnet50", pretrained=True)
        train_ds = ImageViolenceDataset(train_items)
        val_ds = ImageViolenceDataset(val_items) if val_items else None
        train_image_model(model, train_ds, val_ds, epochs=args.epochs)
        logger.info("Image training finished.")

    elif args.cmd == "train-video":
        def read_jsonl(path):
            items = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    j = json.loads(line)
                    items.append({"video_path": j.get("video_path"), "label": int(j.get("label", 0)), "id": j.get("id", None), "start_sec": j.get("start_sec", None)})
            return items
        train_items = read_jsonl(args.train_json)
        val_items = read_jsonl(args.val_json) if args.val_json else None
        model = ViolenceModel(backbone_name="resnet50", pretrained=True)
        train_ds = VideoFrameSequenceDataset(train_items)
        val_ds = VideoFrameSequenceDataset(val_items) if val_items else None
        train_sequence_model(model, train_ds, val_ds, epochs=args.epochs)
        logger.info("Video training finished.")


# -------------------------
# End-of-file guard
# -------------------------
if __name__ == "__main__":
    cli()
