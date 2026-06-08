"""DCGAN Discriminator: 64x64 image -> single logit.

Paper-faithful (SPEC.md §7): strided Conv2d + BatchNorm + LeakyReLU stack, no FC
layers, returns raw logits (use BCEWithLogitsLoss). BatchNorm is omitted on the
first layer per the DCGAN guideline.
"""

import torch
from torch import nn


class Discriminator(nn.Module):
    """Maps a 3x64x64 image to a single real/fake logit of shape (B, 1)."""

    def __init__(self, feature_maps: int = 64, img_channels: int = 3):
        super().__init__()
        fm = feature_maps
        self.net = nn.Sequential(
            # (img_channels, 64, 64) -> (fm, 32, 32); no BatchNorm on the first layer.
            nn.Conv2d(img_channels, fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (fm*2, 16, 16)
            nn.Conv2d(fm, fm * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (fm*4, 8, 8)
            nn.Conv2d(fm * 2, fm * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (fm*8, 4, 4)
            nn.Conv2d(fm * 4, fm * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (1, 1, 1)
            nn.Conv2d(fm * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img).view(img.size(0), 1)
