"""DCGAN weight initialization (SPEC.md §7): conv weights ~ N(0, 0.02)."""

from torch import nn


def init_weights(m: nn.Module) -> None:
    """Apply via ``model.apply(init_weights)``.

    Conv / ConvTranspose weights ~ N(0, 0.02); BatchNorm weights ~ N(1, 0.02),
    biases zeroed — the standard DCGAN scheme.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
