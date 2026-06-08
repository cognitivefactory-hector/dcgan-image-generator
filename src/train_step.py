"""A single alternating DCGAN update: one D-step, then one G-step.

These are the primitives of the custom training loop. M3's ``train.py`` builds the
full epoch loop (checkpointing, fixed-noise logging, FID) around them; keeping the
steps as small, tested functions here is what M1 pins down.

Convention: the Discriminator returns raw logits, so ``criterion`` is
``nn.BCEWithLogitsLoss``. Real -> target 1, fake -> target 0; G is trained to make
D score its fakes as real.
"""

import torch
from torch import nn


def discriminator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    real: torch.Tensor,
    optimizer_d: torch.optim.Optimizer,
    criterion: nn.Module,
    latent_dim: int,
    device: torch.device | str = "cpu",
) -> float:
    """Update D on a real+fake batch. Returns the scalar D loss."""
    real = real.to(device)
    batch_size = real.size(0)

    optimizer_d.zero_grad(set_to_none=True)

    logits_real = discriminator(real)
    loss_real = criterion(logits_real, torch.ones_like(logits_real))

    z = torch.randn(batch_size, latent_dim, device=device)
    fake = generator(z).detach()  # detach: this step trains D only
    logits_fake = discriminator(fake)
    loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))

    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer_d.step()
    return loss_d.item()


def generator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    batch_size: int,
    optimizer_g: torch.optim.Optimizer,
    criterion: nn.Module,
    latent_dim: int,
    device: torch.device | str = "cpu",
) -> float:
    """Update G to fool the current D. Returns the scalar G loss."""
    optimizer_g.zero_grad(set_to_none=True)

    z = torch.randn(batch_size, latent_dim, device=device)
    fake = generator(z)
    logits = discriminator(fake)
    # Target 1: G wants D to score fakes as real.
    loss_g = criterion(logits, torch.ones_like(logits))
    loss_g.backward()
    optimizer_g.step()
    return loss_g.item()
