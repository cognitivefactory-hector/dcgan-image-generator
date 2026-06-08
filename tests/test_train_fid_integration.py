"""M3: train() records a FID value per epoch when a scorer is supplied.

Local-only (needs torchmetrics + Inception); skips in CI like test_fid.py.
"""

import math

import pytest

pytest.importorskip("torchmetrics")

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.eval.fid import FidScorer  # noqa: E402
from src.models.discriminator import Discriminator  # noqa: E402
from src.models.generator import Generator  # noqa: E402
from src.train import train  # noqa: E402

LATENT_DIM = 32


def test_train_appends_finite_fid_per_epoch(tmp_path):
    torch.manual_seed(0)
    images = torch.rand(80, 3, 64, 64) * 2 - 1
    loader = DataLoader(TensorDataset(images, torch.zeros(80, dtype=torch.long)), batch_size=20)

    history = train(
        loader,
        Generator(latent_dim=LATENT_DIM, feature_maps=8),
        Discriminator(feature_maps=8),
        epochs=2,
        latent_dim=LATENT_DIM,
        assets_dir=tmp_path / "assets",
        checkpoint_dir=tmp_path / "checkpoints",
        n_sample=4,
        fid_scorer=FidScorer(feature=64, device="cpu"),
        fid_real_images=images,
        log_fn=lambda _msg: None,
    )

    assert len(history["fid"]) == 2
    assert all(math.isfinite(x) and x >= 0.0 for x in history["fid"])
