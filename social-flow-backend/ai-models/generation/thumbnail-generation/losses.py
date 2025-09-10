"""
Loss functions:
 - pixel L1/L2 loss
 - perceptual loss (VGG) if available
 - adversarial GAN loss helper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLoss(nn.Module):
    def __init__(self, l1=True):
        super().__init__()
        self.l1 = l1
        self.criterion = nn.L1Loss() if l1 else nn.MSELoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)

class VGGLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features from torchvision (if available).
    It extracts a few layers and computes L2 between features.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.to(device)
            self.layers = [3, 8, 15]  # relu1_2, relu2_2, relu3_3 (example)
            self.available = True
        except Exception:
            self.available = False

    def forward(self, pred, target):
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        # normalize to VGG expected input: scale to [0,1] then normalize mean/std
        def prep(x):
            # x: BxCxHxW; scale -> 0-1 already; apply imagenet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
            return (x - mean)/std

        x = prep(pred)
        y = prep(target)
        loss = 0.0
        cur_x = x
        cur_y = y
        for i, layer in enumerate(self.vgg):
            cur_x = layer(cur_x)
            cur_y = layer(cur_y)
            if i in self.layers:
                loss = loss + F.mse_loss(cur_x, cur_y)
        return loss
