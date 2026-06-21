"""EfficientNet-Lite from timm (the reference implementation).

torchvision has no Lite variant; ``tf_efficientnet_lite0`` is the architecture used
by the original DaKanji model, with the same TF-style padding. Same architecture,
so comparable accuracy is expected when trained on the same data. ``lite4`` is the large
end of the sweep: more capacity (and a higher native input resolution), so it favours the
larger ``image_size`` grid points.
"""

from __future__ import annotations

from torch import nn


def _efficientnet_lite(variant: str, num_classes: int, in_channels: int, dropout: float) -> nn.Module:
    import timm

    return timm.create_model(
        variant,
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_channels,
        drop_rate=dropout,
    )


def efficientnet_lite_b0(
    num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object
) -> nn.Module:
    return _efficientnet_lite("tf_efficientnet_lite0", num_classes, in_channels, dropout)


def efficientnet_lite_b4(
    num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object
) -> nn.Module:
    return _efficientnet_lite("tf_efficientnet_lite4", num_classes, in_channels, dropout)
