"""Render the training-story curves to assets/ (PLAN.md M3).

The honest evidence a hiring manager should see: D/G loss curves and the FID
trend over epochs. matplotlib is imported lazily with the non-interactive Agg
backend so this is safe to call headless and keeps the core training loop free of
a plotting dependency.
"""

from pathlib import Path


def _new_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots()


def plot_losses(history: dict[str, list[float]], path: str | Path) -> None:
    """Plot D and G loss per epoch."""
    plt, (fig, ax) = _new_axes()
    epochs = range(len(history["loss_d"]))
    ax.plot(epochs, history["loss_d"], label="D loss")
    ax.plot(epochs, history["loss_g"], label="G loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("DCGAN training losses")
    ax.legend()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fid(history: dict[str, list[float]], path: str | Path) -> None:
    """Plot FID per epoch (the headline metric; lower is better)."""
    plt, (fig, ax) = _new_axes()
    ax.plot(range(len(history["fid"])), history["fid"], marker="o", label="FID")
    ax.set_xlabel("epoch")
    ax.set_ylabel("FID")
    ax.set_title("Fréchet Inception Distance over epochs")
    ax.legend()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
