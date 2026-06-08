"""Nearest-neighbor memorization guard (SPEC.md §7).

For each generated sample, retrieve the closest images in the training set by
pixel-space L2 distance. If generations are near-duplicates of training images
(distance ~ 0), the model is memorizing, not generating. Pair the retrieved pairs
visually in the app/notebook; this module just does the retrieval.
"""

import torch


def nearest_neighbors(
    generated: torch.Tensor,
    references: torch.Tensor,
    k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``k`` nearest references for each generated sample.

    Args:
        generated: ``(N, C, H, W)`` generated images.
        references: ``(M, C, H, W)`` training images to search against.
        k: number of neighbors to return per generated sample.

    Returns:
        ``(distances, indices)``, each ``(N, k)``, sorted nearest-first. ``indices``
        index into ``references``.
    """
    n = generated.size(0)
    m = references.size(0)
    gen_flat = generated.reshape(n, -1)
    ref_flat = references.reshape(m, -1)
    dists = torch.cdist(gen_flat, ref_flat)  # (N, M) Euclidean
    nearest = dists.topk(k, dim=1, largest=False, sorted=True)
    return nearest.values, nearest.indices
