"""M3: the FID wrapper around torchmetrics.

FID needs torchmetrics + an Inception backbone (a ~100MB download) and is too
heavy for CI (PLAN.md "Testing strategy"). Guarded by importorskip so it SKIPS in
CI and runs locally where torchmetrics is installed.
"""

import math

import pytest

pytest.importorskip("torchmetrics")

import torch  # noqa: E402

from src.eval.fid import FidScorer  # noqa: E402


def test_fid_is_finite_nonnegative_and_orders_distributions():
    torch.manual_seed(0)
    # FID needs > `feature` samples for a non-singular covariance; use the small
    # 64-dim head so the test stays cheap on CPU.
    real = torch.rand(80, 3, 64, 64) * 2 - 1  # [-1, 1], our training range
    near = (real + 0.02 * torch.randn_like(real)).clamp(-1, 1)
    far = torch.rand(80, 3, 64, 64) * 2 - 1  # independent distribution

    near_scorer = FidScorer(feature=64, device="cpu")
    near_scorer.add_real(real)
    near_scorer.add_fake(near)
    fid_near = near_scorer.compute()

    far_scorer = FidScorer(feature=64, device="cpu")
    far_scorer.add_real(real)
    far_scorer.add_fake(far)
    fid_far = far_scorer.compute()

    assert math.isfinite(fid_near) and fid_near >= 0.0
    assert math.isfinite(fid_far) and fid_far >= 0.0
    # A near-identical fake distribution must score much closer than an independent one.
    assert fid_near < fid_far
