"""Backbone registry: select an architecture by name from config.

Builders have the signature ``(num_classes, in_channels=1, **kwargs) -> nn.Module``
and ignore kwargs they don't use (``width``/``depth`` are EfficientNet-only).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from char_recognition.models.small_cnn import small_cnn
from char_recognition.models.timm_backbones import efficientnet_lite_b0, efficientnet_lite_b4
from char_recognition.models.tiny_cnn import tiny_cnn
from char_recognition.models.torchvision_backbones import mobilenetv3_large, mobilenetv3_small

BackboneBuilder = Callable[..., nn.Module]

_REGISTRY: dict[str, BackboneBuilder] = {
    "efficientnet_lite_b0": efficientnet_lite_b0,  # timm tf_efficientnet_lite0
    "efficientnet_lite_b4": efficientnet_lite_b4,  # timm tf_efficientnet_lite4 (large end)
    "mobilenetv3_small": mobilenetv3_small,
    "mobilenetv3_large": mobilenetv3_large,
    "small_cnn": small_cnn,  # compact hand-rolled CNN (small end of a real sweep)
    "tiny_cnn": tiny_cnn,  # test-speed baseline, not a deployment candidate
}


def register_backbone(name: str, builder: BackboneBuilder) -> None:
    if name in _REGISTRY:
        raise ValueError(f"backbone {name!r} is already registered")
    _REGISTRY[name] = builder


def available_backbones() -> list[str]:
    return sorted(_REGISTRY)


def build_backbone(name: str, num_classes: int, in_channels: int = 1, **kwargs: object) -> nn.Module:
    try:
        builder = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown backbone {name!r}; available: {available_backbones()}") from exc
    return builder(num_classes, in_channels=in_channels, **kwargs)
