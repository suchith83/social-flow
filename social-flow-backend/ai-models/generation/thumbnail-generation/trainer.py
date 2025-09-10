"""
Training loop for thumbnail generator with optional GAN refinement.

Design:
 - Primary generator trained with pixel + perceptual loss
 - Optionally adversarial refinement: discriminator is trained to distinguish real thumbnails
 - Alternating updates: discriminator step(s) then generator step per batch
 - Multi-aspect training: targets can be different sizes; generator outputs an image at source resolution which is resized to target size for loss computation
"""

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from typing import Optional
from .config import BATCH_SIZE, LR, EPOCHS, DEVICE, LOG_INTERVAL, SAVE_FREQ, USE_GAN, PIXEL_LOSS_WEIGHT, PERCEPTUAL_LOSS_WEIGHT, GAN_LOSS_WEIGHT
from .utils import get_device, set_seed, save_model, compute_image_metrics
from .dataset import collate_batch
from .models import ThumbnailGenerator, PatchDiscriminator
from .losses import PixelLoss, VGGLoss
import os

import logging
logger = logging.getLogger("thumbnail_generation.trainer")

class Trainer:
    def __init__(self, generator: ThumbnailGenerator, discriminator: Optional[PatchDiscriminator] = None, device: str = DEVICE):
        self.device = get_device(device)
        self.gen = generator.to(self.device)
        self.disc = discriminator.to(self.device) if (discriminator is not None and USE_GAN) else None

        self.opt_g = optim.Adam(self.gen.parameters(), lr=LR, betas=(0.5, 0.999))
        if self.disc is not None:
            self.opt_d = optim.Adam(self.disc.parameters(), lr=LR, betas=(0.5, 0.999))

        self.pixel_loss = PixelLoss(l1=True)
        self.percep_loss = VGGLoss(device=self.device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def _disc_step(self, srcs, reals, sizes):
        """
        Train discriminator with real vs fake patches.
        srcs: Bx3xHwxHw (source images resized)
        reals: Bx3xThxTw target thumbnails (already scaled to thumbnail size)
        sizes: list of thumbnail sizes per sample
        We resize src and generated thumbnails to common resolution before concatenation.
        """
        self.disc.train()
        self.opt_d.zero_grad()
        b = srcs.size(0)
        real_scores = []
        fake_scores = []
        losses = []

        for i in range(b):
            size = sizes[i]
            tw, th = size
            # prepare input: resize source to target size to form discriminator input
            src_resized = torch.nn.functional.interpolate(srcs[i:i+1], size=(th, tw), mode="bilinear", align_corners=False)
            real = reals[i:i+1]
            # combine channels
            real_in = torch.cat([src_resized, real], dim=1).to(self.device)
            real_pred = self.disc(real_in)
            real_label = torch.ones_like(real_pred, device=self.device)
            loss_real = self.adversarial_loss(real_pred, real_label)

            # fake generation
            with torch.no_grad():
                gen_full = self.gen(srcs[i:i+1].to(self.device))
                gen_thumb = torch.nn.functional.interpolate(gen_full, size=(th, tw), mode="bilinear", align_corners=False)
            fake_in = torch.cat([src_resized, gen_thumb.detach()], dim=1)
            fake_pred = self.disc(fake_in)
            fake_label = torch.zeros_like(fake_pred, device=self.device)
            loss_fake = self.adversarial_loss(fake_pred, fake_label)

            loss = (loss_real + loss_fake) * 0.5
            losses.append(loss)

        loss_total = torch.stack(losses).mean()
        loss_total.backward()
        self.opt_d.step()
        return loss_total.item()

    def _gen_step(self, srcs, reals, sizes):
        self.gen.train()
        self.opt_g.zero_grad()
        b = srcs.size(0)

        pixel_losses = []
        percep_losses = []
        adv_losses = []

        for i in range(b):
            size = sizes[i]
            tw, th = size
            src = srcs[i:i+1].to(self.device)
            real = reals[i:i+1].to(self.device)

            gen_full = self.gen(src)
            gen_thumb = torch.nn.functional.interpolate(gen_full, size=(th, tw), mode="bilinear", align_corners=False)

            # pixel loss
            l_pix = self.pixel_loss(gen_thumb, real) * PIXEL_LOSS_WEIGHT
            l_perc = self.percep_loss(gen_thumb, real) * PERCEPTUAL_LOSS_WEIGHT if hasattr(self.percep_loss, "available") and self.percep_loss.available else torch.tensor(0.0, device=self.device)

            l_adv = torch.tensor(0.0, device=self.device)
            if self.disc is not None:
                src_resized = torch.nn.functional.interpolate(src, size=(th, tw), mode="bilinear", align_corners=False)
                fake_in = torch.cat([src_resized, gen_thumb], dim=1)
                pred = self.disc(fake_in)
                # generator wants discriminator to predict ones
                l_adv = self.adversarial_loss(pred, torch.ones_like(pred, device=self.device)) * GAN_LOSS_WEIGHT

            total_loss = l_pix + l_perc + l_adv
            total_loss.backward()
            pixel_losses.append(l_pix.item() if isinstance(l_pix, torch.Tensor) else float(l_pix))
            percep_losses.append(l_perc.item() if isinstance(l_perc, torch.Tensor) else float(l_perc))
            adv_losses.append(l_adv.item() if isinstance(l_adv, torch.Tensor) else float(l_adv))

        # step once after accumulating gradients across batch (already backwarded per sample)
        self.opt_g.step()
        # zero grad already done at top next iteration
        stats = {
            "pixel_loss": float(sum(pixel_losses) / max(len(pixel_losses),1)),
            "percep_loss": float(sum(percep_losses) / max(len(percep_losses),1)),
            "adv_loss": float(sum(adv_losses) / max(len(adv_losses),1))
        }
        return stats

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = EPOCHS):
        set_seed()
        device = self.device
        for epoch in range(1, epochs + 1):
            self.gen.train()
            if self.disc: self.disc.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
            for step, batch in pbar:
                srcs, targets, paths, boxes, sizes = batch
                # move to device later inside steps
                if self.disc is not None:
                    d_loss = self._disc_step(srcs, targets, sizes)
                else:
                    d_loss = 0.0

                g_stats = self._gen_step(srcs, targets, sizes)

                if step % LOG_INTERVAL == 0:
                    pbar.set_postfix({"d_loss": d_loss, **g_stats})

            # epoch end: checkpoint
            if epoch % SAVE_FREQ == 0:
                gen_path = os.path.join("models", f"generator_epoch{epoch}.pth")
                torch.save(self.gen.state_dict(), gen_path)
                logger.info(f"Saved generator to {gen_path}")
                if self.disc is not None:
                    disc_path = os.path.join("models", f"discriminator_epoch{epoch}.pth")
                    torch.save(self.disc.state_dict(), disc_path)
                    logger.info(f"Saved discriminator to {disc_path}")

            # optionally run validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                logger.info(f"Validation metrics epoch {epoch}: {val_metrics}")

    def validate(self, val_loader: DataLoader, num_samples: int = 32):
        self.gen.eval()
        if self.disc: self.disc.eval()
        metrics = {"psnr": [], "ssim": []}
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                srcs, targets, paths, boxes, sizes = batch
                srcs = srcs.to(self.device)
                targets = targets.to(self.device)
                gen_full = self.gen(srcs)
                # resize generated to target size for metric computation
                gen_resized = torch.nn.functional.interpolate(gen_full, size=(targets.shape[2], targets.shape[3]), mode="bilinear", align_corners=False)
                stat = compute_image_metrics(gen_resized, targets)
                metrics["psnr"].append(stat["psnr"])
                metrics["ssim"].append(stat["ssim"])
                if step * val_loader.batch_size >= num_samples:
                    break
        out = {"psnr": float(sum(metrics["psnr"])/len(metrics["psnr"])), "ssim": float(sum(metrics["ssim"])/len(metrics["ssim"]))}
        return out
