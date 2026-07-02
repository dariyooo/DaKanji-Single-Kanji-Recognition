"""Augmentation via ``torchvision.transforms.v2``.

Per-sample: shear, sharpness, stroke-width, blur, cutout. Batch-level: MixUp (in the collate
fn, producing soft labels). Images stay float ``[0, 255]``; the in-model ``Preprocess`` does
the final normalize.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from torchvision.transforms import v2

from char_recognition.config.augment import AugmentConfig

Batch = tuple[Tensor, Tensor]


class RandomStrokeWidth(nn.Module):
    """Randomly thin or thicken strokes via morphological min/max pooling.

    Spans the thin-handwriting vs thick-rendered-data gap so the model is robust to both.
    Polarity-agnostic: half the time it expands light regions (thins dark ink), half the time
    it expands dark regions (thickens dark ink), covering both directions regardless of which
    is foreground. ``p`` is the total probability of applying either.
    """

    def __init__(self, p: float = 0.0, kernel_size: int = 3) -> None:
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size

    def forward(self, image: Tensor) -> Tensor:
        roll = float(torch.rand(()))
        if roll >= self.p:
            return image
        pad = self.kernel_size // 2
        x = image.unsqueeze(0)  # (1, C, H, W) for pooling
        if roll < self.p / 2:  # local max: thin dark strokes
            x = F.max_pool2d(x, self.kernel_size, stride=1, padding=pad)
        else:  # local min: thicken dark strokes
            x = -F.max_pool2d(-x, self.kernel_size, stride=1, padding=pad)
        return x.squeeze(0)


def build_train_transform(cfg: AugmentConfig) -> Callable[[Tensor], Tensor]:
    """Per-sample transform on a ``(C, H, W)`` float tensor."""
    return v2.Compose(
        [
            v2.RandomAffine(degrees=0, shear=cfg.shear_degrees, fill=0.0),
            v2.RandomAdjustSharpness(sharpness_factor=cfg.sharpness_factor, p=cfg.sharpness_p),
            RandomStrokeWidth(p=cfg.stroke_p),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=cfg.blur_sigma)], p=cfg.blur_p),
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
    """Build a :class:`MixCollate`, or ``None`` when mixing is disabled.

    MixUp only by default; CutMix is gated behind ``use_cutmix`` because halving a glyph
    destroys its identity (bad for single-character recognition).
    """
    if cfg.mix_p <= 0.0:
        return None
    mixup = v2.MixUp(num_classes=num_classes, alpha=cfg.mixup_alpha)
    if cfg.use_cutmix:
        mix: Callable[[Tensor, Tensor], Batch] = v2.RandomChoice(
            [v2.CutMix(num_classes=num_classes, alpha=cfg.cutmix_alpha), mixup]
        )
    else:
        mix = mixup
    return MixCollate(mix, cfg.mix_p)
