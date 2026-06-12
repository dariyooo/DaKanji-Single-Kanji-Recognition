"""Augmentation via ``torchvision.transforms.v2``.

Per-sample: shear, sharpness, cutout. Batch-level: MixUp / CutMix (in the collate
fn, producing soft labels). Images stay float ``[0, 255]``; the in-model
``Preprocess`` does the final normalize.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import default_collate
from torchvision.transforms import v2

from char_recognition.config import AugmentConfig

Batch = tuple[Tensor, Tensor]


def build_train_transform(cfg: AugmentConfig) -> Callable[[Tensor], Tensor]:
    """Per-sample transform on a ``(C, H, W)`` float tensor."""
    return v2.Compose(
        [
            v2.RandomAffine(degrees=0, shear=cfg.shear_degrees, fill=0.0),
            v2.RandomAdjustSharpness(sharpness_factor=cfg.sharpness_factor, p=cfg.sharpness_p),
            v2.RandomErasing(p=cfg.cutout_p, scale=cfg.cutout_scale, ratio=cfg.cutout_ratio, value=0.0),
        ]
    )


class MixCollate:
    """Collate that applies MixUp/CutMix to a batch.

    A module-level class (not a closure) so it pickles for DataLoader workers under
    the ``spawn`` start method (the default on macOS/Windows). When mixing fires,
    labels become soft ``(B, num_classes)`` targets; otherwise they stay int class
    indices, and cross-entropy accepts both, so batches may differ.
    """

    def __init__(self, mix: Callable[[Tensor, Tensor], Batch], mix_p: float) -> None:
        self.mix = mix
        self.mix_p = mix_p

    def __call__(self, batch: list) -> Batch:
        images, labels = default_collate(batch)
        if images.shape[0] > 1 and torch.rand(()) < self.mix_p:
            images, labels = self.mix(images, labels)
        return images, labels


def build_mix_collate(num_classes: int, cfg: AugmentConfig) -> MixCollate | None:
    """Build a :class:`MixCollate`, or ``None`` when mixing is disabled."""
    if cfg.mix_p <= 0.0:
        return None
    mix = v2.RandomChoice(
        [
            v2.CutMix(num_classes=num_classes, alpha=cfg.cutmix_alpha),
            v2.MixUp(num_classes=num_classes, alpha=cfg.mixup_alpha),
        ]
    )
    return MixCollate(mix, cfg.mix_p)
