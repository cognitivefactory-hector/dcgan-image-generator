"""Fréchet Inception Distance, the headline metric (SPEC.md §7).

Thin wrapper over torchmetrics' ``FrechetInceptionDistance``. FID compares the
Inception feature statistics of generated vs. real images; tracked over epochs it
is the honest quality signal (trending down) — paired with the fixed-noise grid,
since FID alone doesn't measure diversity directly.

torchmetrics is an optional/heavy dependency (Inception backbone download), so it
is imported lazily: importing this module is cheap; constructing ``FidScorer``
pulls torchmetrics in.

Convention: images are passed in the training range [-1, 1] and converted to the
[0, 1] that the metric expects (``normalize=True``).
"""

import torch


class FidScorer:
    """Accumulate real and generated images, then compute FID.

    Typical use per epoch: add the real set once, add the epoch's fakes, compute,
    then ``reset`` the fake side for the next epoch (or build a fresh scorer).
    """

    def __init__(self, feature: int = 2048, device: torch.device | str = "cpu"):
        # Lazy import: keeps `import src.eval.fid` free of the heavy dependency.
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.device = device
        self.metric = FrechetInceptionDistance(feature=feature, normalize=True).to(device)

    @staticmethod
    def _to_unit(images: torch.Tensor) -> torch.Tensor:
        """Map [-1, 1] -> [0, 1] for the metric (normalize=True expects floats)."""
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)

    def add_real(self, images: torch.Tensor) -> None:
        self.metric.update(self._to_unit(images).to(self.device), real=True)

    def add_fake(self, images: torch.Tensor) -> None:
        self.metric.update(self._to_unit(images).to(self.device), real=False)

    def compute(self) -> float:
        return self.metric.compute().item()

    def reset(self) -> None:
        self.metric.reset()
