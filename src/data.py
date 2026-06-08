"""Data pipeline: load images and normalize to the DCGAN training range.

Transforms (SPEC.md §5): resize + center-crop to ``image_size`` (default 64, to
match the M1 models), then map pixels to [-1, 1] so they align with the
Generator's ``tanh`` output.

Dataset provenance (the dataset is **not** committed — see .gitignore):
arrange a public image dataset (e.g. anime faces / CelebA / flowers; state its
license in the README) as a torchvision ``ImageFolder`` under ``data/``::

    data/
      <any_class_name>/
        img0.jpg
        img1.jpg
        ...

The class label is unused (GAN training is unconditional) but ImageFolder still
requires at least one class subdirectory. Document the fetch step in the README.
"""

from os import PathLike

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 64) -> transforms.Compose:
    """Resize/center-crop to ``image_size`` and normalize RGB to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
        ]
    )


def make_dataloader(
    data_dir: str | PathLike,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader over an ImageFolder at ``data_dir``.

    ``drop_last=True`` keeps every batch full (BatchNorm prefers it); ``pin_memory``
    is enabled only when a CUDA device is present, to avoid a CPU-only warning.
    """
    dataset = datasets.ImageFolder(root=str(data_dir), transform=build_transforms(image_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
