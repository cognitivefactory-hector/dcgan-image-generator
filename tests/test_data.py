"""M2: the data pipeline produces correctly shaped, [-1, 1]-normalized batches.

Uses images generated into a tmp dir (no downloaded dataset), so it runs in CI.
See PLAN.md M2 and SPEC.md §5.
"""

import torch
from PIL import Image

from src.data import build_transforms, make_dataloader

IMG_SIZE = 64


def test_transform_outputs_normalized_chw_tensor():
    tf = build_transforms(image_size=IMG_SIZE)
    out = tf(Image.new("RGB", (128, 96), color=(255, 0, 0)))
    assert out.shape == (3, IMG_SIZE, IMG_SIZE)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_transform_maps_pixel_extremes_to_pm1():
    tf = build_transforms(image_size=IMG_SIZE)
    black = tf(Image.new("RGB", (64, 64), (0, 0, 0)))
    white = tf(Image.new("RGB", (64, 64), (255, 255, 255)))
    assert torch.allclose(black.min(), torch.tensor(-1.0), atol=1e-6)
    assert torch.allclose(white.max(), torch.tensor(1.0), atol=1e-6)


def test_dataloader_yields_expected_batch(tmp_path):
    # ImageFolder expects class subdirectories.
    class_dir = tmp_path / "faces"
    class_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (80, 80), (i * 30, i * 30, i * 30)).save(class_dir / f"{i}.png")

    loader = make_dataloader(tmp_path, image_size=IMG_SIZE, batch_size=2, num_workers=0)
    images, _ = next(iter(loader))

    assert images.shape == (2, 3, IMG_SIZE, IMG_SIZE)
    assert images.dtype == torch.float32
    assert images.min() >= -1.0
    assert images.max() <= 1.0
