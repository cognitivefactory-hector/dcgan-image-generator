"""M1 mechanics: DCGAN weight init draws conv weights from N(0, 0.02)."""

import torch

from src.models.generator import Generator
from src.models.weights import init_weights


def test_init_weights_sets_conv_std_near_002():
    torch.manual_seed(0)
    g = Generator(latent_dim=100, feature_maps=64)
    g.apply(init_weights)
    conv_weights = torch.cat(
        [
            m.weight.flatten()
            for m in g.modules()
            if isinstance(m, torch.nn.ConvTranspose2d)
        ]
    )
    # N(0, 0.02): mean ~0, std ~0.02. Loose bounds — it's a stochastic draw.
    assert abs(conv_weights.mean().item()) < 0.005
    assert abs(conv_weights.std().item() - 0.02) < 0.005
