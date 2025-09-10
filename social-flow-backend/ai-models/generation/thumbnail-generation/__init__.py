"""
Thumbnail Generation Package

Provides models, trainers, data utilities and pipelines for generating thumbnails
from source images/videos. Supports:
 - deterministic encoder-decoder (MSE/ perceptual loss)
 - adversarial refinement (GAN discriminator)
 - smart crop suggestions
 - evaluation: PSNR, SSIM, LPIPS (if available)
"""

__version__ = "1.0.0"
__author__ = "AI Thumbnail Team"
