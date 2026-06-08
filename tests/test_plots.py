"""M3: loss/FID curve plotting saves PNGs to assets/.

matplotlib is not installed in CI (kept lean), so this is guarded and skips there.
"""

import pytest

pytest.importorskip("matplotlib")

from src.eval.plots import plot_fid, plot_losses  # noqa: E402


def test_plot_losses_writes_png(tmp_path):
    history = {"loss_d": [1.2, 0.9, 0.7], "loss_g": [2.0, 1.6, 1.4], "fid": []}
    out = tmp_path / "loss_curve.png"
    plot_losses(history, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_fid_writes_png(tmp_path):
    history = {"loss_d": [], "loss_g": [], "fid": [180.0, 120.0, 90.0]}
    out = tmp_path / "fid_curve.png"
    plot_fid(history, out)
    assert out.exists() and out.stat().st_size > 0
