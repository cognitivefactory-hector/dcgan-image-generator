"""Custom DCGAN training loop with honest instrumentation (PLAN.md M3 — the heart).

Each epoch: alternate D/G steps over the data, record mean D/G losses, write a
**fixed-noise sample grid** (same noise every epoch, so diversity loss = mode
collapse is visible at a glance), optionally compute **FID**, and checkpoint. The
loss/FID history is written to ``assets/history.json``; curve plots are rendered
separately by ``src.eval.plots`` (kept out of this module so the core loop has no
matplotlib dependency).

Training is offline (GPU; SPEC.md §9). This module is import-light on purpose:
it pulls only torch + torchvision, so the loop is unit-testable on CPU.
"""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import make_grid, save_image

from src.models.weights import init_weights
from src.train_step import discriminator_step, generator_step


def _denorm(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for saving/grids."""
    return (images + 1.0) / 2.0


@torch.no_grad()
def _save_sample_grid(generator: nn.Module, fixed_noise: torch.Tensor, path: Path) -> None:
    was_training = generator.training
    generator.eval()
    fake = generator(fixed_noise)
    if was_training:
        generator.train()
    nrow = max(1, int(fixed_noise.size(0) ** 0.5))
    save_image(make_grid(_denorm(fake), nrow=nrow), str(path))


@torch.no_grad()
def _sample(generator: nn.Module, n: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    was_training = generator.training
    generator.eval()
    fakes = generator(torch.randn(n, latent_dim, device=device))
    if was_training:
        generator.train()
    return fakes


def train(
    dataloader,
    generator: nn.Module,
    discriminator: nn.Module,
    *,
    epochs: int,
    latent_dim: int,
    device: torch.device | str = "cpu",
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    assets_dir: str | Path = "assets",
    checkpoint_dir: str | Path = "checkpoints",
    fixed_noise: torch.Tensor | None = None,
    n_sample: int = 64,
    fid_scorer=None,
    fid_real_images: torch.Tensor | None = None,
    init: bool = True,
    log_fn: Callable[[str], None] = print,
) -> dict[str, list[float]]:
    """Run the alternating DCGAN training loop and save honest-evaluation artifacts.

    Args:
        dataloader: yields ``(images, _)`` with images in [-1, 1].
        epochs, latent_dim, lr, betas: training hyperparameters (DCGAN defaults).
        assets_dir: per-epoch fixed-noise grids + ``history.json`` land here.
        checkpoint_dir: ``last.pt`` (G/D state + epoch + latent_dim) is rewritten each epoch.
        fixed_noise: reuse the SAME noise across epochs for the progression grid;
            generated if omitted.
        fid_scorer, fid_real_images: if both given, FID is computed each epoch.
        init: apply N(0, 0.02) DCGAN weight init before training.

    Returns:
        ``{"loss_d": [...], "loss_g": [...], "fid": [...]}`` (``fid`` empty if disabled).
    """
    device = torch.device(device)
    generator.to(device)
    discriminator.to(device)
    if init:
        generator.apply(init_weights)
        discriminator.apply(init_weights)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    criterion = nn.BCEWithLogitsLoss()

    if fixed_noise is None:
        fixed_noise = torch.randn(n_sample, latent_dim, device=device)
    else:
        fixed_noise = fixed_noise.to(device)

    assets_dir = Path(assets_dir)
    checkpoint_dir = Path(checkpoint_dir)
    grids_dir = assets_dir / "fixed_noise"
    grids_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"loss_d": [], "loss_g": [], "fid": []}

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        loss_d_sum = loss_g_sum = 0.0
        n_batches = 0
        for real, _ in dataloader:
            real = real.to(device)
            loss_d = discriminator_step(
                discriminator, generator, real, opt_d, criterion, latent_dim, device
            )
            loss_g = generator_step(
                discriminator, generator, real.size(0), opt_g, criterion, latent_dim, device
            )
            loss_d_sum += loss_d
            loss_g_sum += loss_g
            n_batches += 1

        n_batches = max(n_batches, 1)
        history["loss_d"].append(loss_d_sum / n_batches)
        history["loss_g"].append(loss_g_sum / n_batches)

        _save_sample_grid(generator, fixed_noise, grids_dir / f"epoch_{epoch:03d}.png")

        if fid_scorer is not None and fid_real_images is not None:
            fid_scorer.reset()
            fid_scorer.add_real(fid_real_images)
            fid_scorer.add_fake(_sample(generator, fid_real_images.size(0), latent_dim, device))
            history["fid"].append(fid_scorer.compute())

        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "latent_dim": latent_dim,
            },
            checkpoint_dir / "last.pt",
        )

        fid_msg = f" fid={history['fid'][-1]:.2f}" if history["fid"] else ""
        log_fn(
            f"epoch {epoch}: loss_d={history['loss_d'][-1]:.3f} "
            f"loss_g={history['loss_g'][-1]:.3f}{fid_msg}"
        )

    (assets_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def _collect_real(dataloader, n: int) -> torch.Tensor:
    """Gather a fixed set of ``n`` real images for the FID 'real' side."""
    batches = []
    count = 0
    for images, _ in dataloader:
        batches.append(images)
        count += images.size(0)
        if count >= n:
            break
    return torch.cat(batches)[:n]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for offline training (GPU recommended; SPEC.md §9).

    Example:
        python -m src.train --data-dir data --epochs 50 --fid
    """
    import argparse

    from src.data import make_dataloader
    from src.models.discriminator import Discriminator
    from src.models.generator import Generator

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Train the DCGAN (offline; GPU recommended).")
    parser.add_argument("--data-dir", required=True, help="ImageFolder root (see src/data.py).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--feature-maps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--fid", action="store_true", help="compute FID each epoch")
    parser.add_argument("--fid-samples", type=int, default=2048)
    args = parser.parse_args(argv)

    loader = make_dataloader(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    generator = Generator(latent_dim=args.latent_dim, feature_maps=args.feature_maps)
    discriminator = Discriminator(feature_maps=args.feature_maps)

    fid_scorer = None
    fid_real_images = None
    if args.fid:
        from src.eval.fid import FidScorer

        fid_scorer = FidScorer(device=args.device)
        fid_real_images = _collect_real(loader, args.fid_samples)

    history = train(
        loader,
        generator,
        discriminator,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        device=args.device,
        lr=args.lr,
        assets_dir=args.assets_dir,
        checkpoint_dir=args.checkpoint_dir,
        fid_scorer=fid_scorer,
        fid_real_images=fid_real_images,
    )

    try:
        from src.eval.plots import plot_fid, plot_losses

        plot_losses(history, Path(args.assets_dir) / "loss_curve.png")
        if history["fid"]:
            plot_fid(history, Path(args.assets_dir) / "fid_curve.png")
    except ImportError:
        print("matplotlib not installed; skipping curve plots (history.json still saved).")


if __name__ == "__main__":
    main()
