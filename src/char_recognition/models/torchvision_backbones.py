"""MobileNetV3 (torchvision) adapted to grayscale: first conv swapped to ``in_channels``."""

from __future__ import annotations

from typing import cast

from torch import nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

__all__ = ["mobilenetv3_large", "mobilenetv3_small"]


def _adapt_first_conv(model: nn.Module, in_channels: int) -> None:
    stem = cast(nn.Sequential, cast(nn.Sequential, model.features)[0])
    first = cast(nn.Conv2d, stem[0])
    if first.in_channels == in_channels:
        return
    stem[0] = nn.Conv2d(
        in_channels,
        first.out_channels,
        kernel_size=cast("tuple[int, int]", first.kernel_size),
        stride=cast("tuple[int, int]", first.stride),
        padding=cast("str | tuple[int, int]", first.padding),
        bias=first.bias is not None,
    )


def mobilenetv3_small(
    num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object
) -> nn.Module:
    model = mobilenet_v3_small(weights=None, num_classes=num_classes, dropout=dropout)
    _adapt_first_conv(model, in_channels)
    return model


def mobilenetv3_large(
    num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object
) -> nn.Module:
    model = mobilenet_v3_large(weights=None, num_classes=num_classes, dropout=dropout)
    _adapt_first_conv(model, in_channels)
    return model
