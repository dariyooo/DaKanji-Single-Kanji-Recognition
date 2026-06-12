"""EfficientNet-Lite from timm (the reference implementation).

torchvision has no Lite variant; ``tf_efficientnet_lite0`` is the architecture used
by the original DaKanji model, with the same TF-style padding. Same architecture,
so comparable accuracy is expected when trained on the same data.
"""

from __future__ import annotations

from torch import nn


def efficientnet_lite_b0(
    num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object
) -> nn.Module:
    import timm

    return timm.create_model(
        "tf_efficientnet_lite0",
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_channels,
        drop_rate=dropout,
    )
