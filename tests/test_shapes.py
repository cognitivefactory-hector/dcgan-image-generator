"""M1 mechanics: Generator / Discriminator I/O shapes and DCGAN invariants.

Lightweight (CPU, tiny feature maps) so these run in CI. See PLAN.md M1.
"""

import torch

from src.models.discriminator import Discriminator
from src.models.generator import Generator

LATENT_DIM = 100
IMG_SIZE = 64


def test_generator_maps_noise_to_image_batch():
    g = Generator(latent_dim=LATENT_DIM, feature_maps=8)
    z = torch.randn(2, LATENT_DIM)
    out = g(z)
    assert out.shape == (2, 3, IMG_SIZE, IMG_SIZE)


def test_generator_output_is_tanh_bounded():
    g = Generator(latent_dim=LATENT_DIM, feature_maps=8)
    out = g(torch.randn(4, LATENT_DIM))
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_discriminator_maps_image_to_single_logit():
    d = Discriminator(feature_maps=8)
    img = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out = d(img)
    assert out.shape == (2, 1)


def test_discriminator_has_no_fully_connected_layers():
    # DCGAN guideline: all-convolutional, no nn.Linear anywhere.
    d = Discriminator(feature_maps=8)
    assert not any(isinstance(m, torch.nn.Linear) for m in d.modules())
