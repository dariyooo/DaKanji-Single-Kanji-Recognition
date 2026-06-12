"""Augmentation strengths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AugmentConfig:
    shear_degrees: float = 10.0
    sharpness_factor: float = 2.0
    sharpness_p: float = 0.3
    cutout_p: float = 0.3
    cutout_scale: tuple[float, float] = (0.02, 0.4)
    cutout_ratio: tuple[float, float] = (0.3, 3.3)
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_p: float = 0.5  # 0 disables MixUp/CutMix
