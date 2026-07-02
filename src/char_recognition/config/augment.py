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
    # Stroke-width robustness (thin handwriting vs thick rendered data). Off by default;
    # enable per dataset (e.g. the kanji run config).
    stroke_p: float = 0.0  # randomly thin/thicken strokes (morphological min/max pool)
    blur_p: float = 0.0  # randomly soften strokes (Gaussian blur)
    blur_sigma: tuple[float, float] = (0.1, 1.5)
    # Batch-level mixing. CutMix halves a glyph (destroys its identity), so it's off for
    # single-character recognition; MixUp (a global blend) keeps the glyph whole.
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    use_cutmix: bool = False
    mix_p: float = 0.2  # 0 disables mixing
