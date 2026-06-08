"""M3: the nearest-neighbor memorization guard retrieves closest training images."""

import torch

from src.eval.nearest_neighbor import nearest_neighbors


def test_returns_distances_and_indices_with_expected_shapes():
    generated = torch.randn(3, 3, 8, 8)
    references = torch.randn(5, 3, 8, 8)
    distances, indices = nearest_neighbors(generated, references, k=2)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2)


def test_exact_copy_has_zero_distance_to_its_source():
    references = torch.randn(5, 3, 8, 8)
    # A generated sample that is an exact copy of reference #3.
    generated = references[3:4].clone()
    distances, indices = nearest_neighbors(generated, references, k=1)
    assert indices[0, 0].item() == 3
    assert torch.allclose(distances[0, 0], torch.tensor(0.0), atol=1e-5)


def test_neighbors_are_sorted_nearest_first():
    references = torch.randn(6, 3, 8, 8)
    generated = torch.randn(2, 3, 8, 8)
    distances, _ = nearest_neighbors(generated, references, k=3)
    # Each row is non-decreasing.
    assert torch.all(distances[:, 1:] >= distances[:, :-1])
