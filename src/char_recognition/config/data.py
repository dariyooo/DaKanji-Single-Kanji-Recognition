"""Dataset location and batching."""

from __future__ import annotations

from dataclasses import dataclass, field

from char_recognition.paths import LABELS_FILE


@dataclass
class DataConfig:
    """Dataset location and batching. ``root=None`` uses synthetic data."""

    root: str | None = None
    val_root: str | None = None  # held-out set for final evaluation (scripts/evaluate.py)
    labels_file: str = field(default_factory=lambda: str(LABELS_FILE))
    synthetic_classes: int = 10  # class count when root is None
    image_size: tuple[int, int] = (64, 64)
    in_channels: int = 1
    val_split: float = 0.2  # in-training train/val split of ``root``
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    seed: int = 123
