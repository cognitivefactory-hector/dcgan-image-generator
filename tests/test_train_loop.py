"""M3: the custom training loop wiring — runs end-to-end and saves artifacts.

This is a *mechanics* smoke (tiny models, 6 random images, 1 epoch, CPU): it checks
that the loop alternates D/G, records losses, writes a fixed-noise grid, a
checkpoint, and history.json. It does NOT check sample quality — that's FID + the
offline GPU run (PLAN.md M3 acceptance).
"""

import json
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.train import train

LATENT_DIM = 32


def _tiny_loader():
    images = torch.rand(6, 3, 64, 64) * 2 - 1  # [-1, 1]
    labels = torch.zeros(6, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=3)


def test_train_runs_and_saves_artifacts(tmp_path):
    g = Generator(latent_dim=LATENT_DIM, feature_maps=8)
    d = Discriminator(feature_maps=8)
    assets = tmp_path / "assets"
    ckpts = tmp_path / "checkpoints"

    history = train(
        _tiny_loader(),
        g,
        d,
        epochs=2,
        latent_dim=LATENT_DIM,
        assets_dir=assets,
        checkpoint_dir=ckpts,
        n_sample=4,
        log_fn=lambda _msg: None,
    )

    # Loss history: one finite entry per epoch.
    assert len(history["loss_d"]) == 2
    assert len(history["loss_g"]) == 2
    assert all(math.isfinite(x) for x in history["loss_d"] + history["loss_g"])

    # Artifacts on disk.
    assert (ckpts / "last.pt").exists()
    assert (assets / "fixed_noise" / "epoch_000.png").exists()
    assert (assets / "fixed_noise" / "epoch_001.png").exists()
    assert (assets / "history.json").exists()
    saved = json.loads((assets / "history.json").read_text())
    assert saved["loss_d"] == history["loss_d"]


def test_checkpoint_roundtrips_into_a_fresh_generator(tmp_path):
    g = Generator(latent_dim=LATENT_DIM, feature_maps=8)
    d = Discriminator(feature_maps=8)
    train(
        _tiny_loader(),
        g,
        d,
        epochs=1,
        latent_dim=LATENT_DIM,
        assets_dir=tmp_path / "assets",
        checkpoint_dir=tmp_path / "checkpoints",
        n_sample=4,
        log_fn=lambda _msg: None,
    )

    ckpt = torch.load(tmp_path / "checkpoints" / "last.pt")
    fresh = Generator(latent_dim=LATENT_DIM, feature_maps=8)
    fresh.load_state_dict(ckpt["generator"])  # must match the saved architecture
    assert ckpt["latent_dim"] == LATENT_DIM
