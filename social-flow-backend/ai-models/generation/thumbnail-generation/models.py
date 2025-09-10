"""
Model definitions:

- Encoder: standard conv encoder (resnet-inspired, small)
- Decoder: upsampling decoder (UNet-like) that produces thumbnails at requested sizes
- Optional Discriminator for adversarial refinement (PatchGAN)
- Perceptual loss helper: use VGG features for perceptual loss if torchvision available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from .config import ENCODER_FEATURES, DECODER_FEATURES, LATENT_DIM, USE_GAN

# -------------------------
# Encoder
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    """
    Simple hierarchical encoder that downsamples to high-level features.
    Input: Bx3xH_xW (SRC_SIZE)
    Output: dictionary of encoder feature maps for skip connections + bottleneck feature
    """
    def __init__(self, in_channels=3, features: List[int] = ENCODER_FEATURES):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.layers.append(nn.Sequential(
                ConvBlock(ch, f),
                ConvBlock(f, f),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            ch = f

    def forward(self, x):
        feats = []
        cur = x
        for layer in self.layers:
            cur = layer(cur)
            feats.append(cur)
        # feats[-1] is deepest map
        return feats

# -------------------------
# Decoder
# -------------------------
class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock(out_ch*2, out_ch),
            ConvBlock(out_ch, out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # if sizes do not match due to rounding, center crop skip
        if x.shape != skip.shape:
            # center crop skip to x shape
            _,_,h,w = x.shape
            skip = center_crop(skip, (h,w))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

def center_crop(tensor, size):
    _,_,h,w = tensor.shape
    new_h, new_w = size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return tensor[:,:,top:top+new_h, left:left+new_w]

class Decoder(nn.Module):
    """
    UNet-style decoder that maps bottleneck to full-resolution feature maps.
    We'll provide a small head that can produce outputs at different sizes by adaptive pooling and convs.
    """
    def __init__(self, features: List[int] = DECODER_FEATURES, out_channels=3):
        super().__init__()
        # symmetrical design: the deepest encoder feature should match first decoder in channels
        self.up_blocks = nn.ModuleList()
        for i in range(len(features)-1):
            self.up_blocks.append(UpConvBlock(features[i], features[i+1]))
        # head
        self.head = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]//2, out_channels, kernel_size=1)
        )

    def forward(self, bottleneck, skips):
        """
        bottleneck: deepest-level feature map
        skips: list of skip feature maps from encoder (in reverse order)
        """
        x = bottleneck
        for i, up in enumerate(self.up_blocks):
            # skip order: provide corresponding
            skip = skips[i] if i < len(skips) else None
            if skip is None:
                # upsample without skip
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = up.conv(x, x) if hasattr(up, "conv") else x
            else:
                x = up(x, skip)
        out = torch.sigmoid(self.head(x))  # output in [0,1]
        return out

# -------------------------
# Full Generator wrapper
# -------------------------
class ThumbnailGenerator(nn.Module):
    """
    Full encoder-decoder generator. Produces an output map at source resolution.
    To obtain thumbnails at different sizes, we apply adaptive pooling/resizing on the output.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        # The encoder features list defines depth; pick last encoder channels as bottleneck channel
        bottleneck_ch = ENCODER_FEATURES[-1]
        # Create a simple 1x1 conv bottleneck transform to feed decoder
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, DECODER_FEATURES[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = Decoder(features=DECODER_FEATURES, out_channels=out_channels)

    def forward(self, x):
        feats = self.encoder(x)
        # prepare skips reversed except deepest
        # feats: [lvl1, lvl2, lvl3, lvl4], we want skips = [lvl3, lvl2, lvl1] for decoder
        skips = feats[:-1][::-1]
        bott = feats[-1]
        bott = self.bottleneck(bott)
        out = self.decoder(bott, skips)
        return out

# -------------------------
# Discriminator (PatchGAN)
# -------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, base_features=64):
        """
        Input: concatenated (source_image, generated_or_real_thumbnail) resized to same resolution
        """
        super().__init__()
        layers = []
        ch = in_channels
        f = base_features
        for i in range(4):
            layers.append(nn.Conv2d(ch, f, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = f
            f = min(f*2, 512)
        layers.append(nn.Conv2d(ch, 1, kernel_size=4, padding=1))  # patch output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
