"""DCGAN Generator: noise z -> 64x64 image.

Paper-faithful (SPEC.md §7): project/reshape z, then a stack of fractional-strided
ConvTranspose2d + BatchNorm + ReLU, with a tanh output in [-1, 1]. No FC layers.
"""

import torch
from torch import nn


class Generator(nn.Module):
    """Maps a latent vector to a 3x64x64 image.

    Accepts noise shaped (B, latent_dim) or (B, latent_dim, 1, 1); it is reshaped
    to (B, latent_dim, 1, 1) before the transposed-conv stack.
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, img_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        fm = feature_maps
        self.net = nn.Sequential(
            # (latent_dim, 1, 1) -> (fm*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, fm * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.ReLU(inplace=True),
            # -> (fm*4, 8, 8)
            nn.ConvTranspose2d(fm * 8, fm * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fm * 4),
            nn.ReLU(inplace=True),
            # -> (fm*2, 16, 16)
            nn.ConvTranspose2d(fm * 4, fm * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fm * 2),
            nn.ReLU(inplace=True),
            # -> (fm, 32, 32)
            nn.ConvTranspose2d(fm * 2, fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(inplace=True),
            # -> (img_channels, 64, 64)
            nn.ConvTranspose2d(fm, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)
