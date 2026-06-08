"""M1 mechanics: one alternating D-step + G-step runs and yields finite losses.

This tests the *mechanics* of a single update (gradients flow, optimizers step,
losses are finite) — not sample quality. See PLAN.md M1 and "Testing strategy".
"""

import math

import torch
from torch import nn

from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.train_step import discriminator_step, generator_step

LATENT_DIM = 100


def _setup():
    torch.manual_seed(0)
    g = Generator(latent_dim=LATENT_DIM, feature_maps=8)
    d = Discriminator(feature_maps=8)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    return g, d, opt_g, opt_d, criterion


def test_discriminator_step_returns_finite_loss():
    g, d, _, opt_d, criterion = _setup()
    real = torch.randn(4, 3, 64, 64)
    loss_d = discriminator_step(d, g, real, opt_d, criterion, LATENT_DIM)
    assert math.isfinite(loss_d)


def test_generator_step_returns_finite_loss():
    g, d, opt_g, _, criterion = _setup()
    loss_g = generator_step(d, g, 4, opt_g, criterion, LATENT_DIM)
    assert math.isfinite(loss_g)


def test_generator_step_updates_generator_weights():
    # A real gradient step must change G's parameters.
    g, d, opt_g, _, criterion = _setup()
    before = next(g.parameters()).clone()
    generator_step(d, g, 4, opt_g, criterion, LATENT_DIM)
    after = next(g.parameters())
    assert not torch.equal(before, after)
