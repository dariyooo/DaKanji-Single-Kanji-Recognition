"""Optimizer, schedule and mixed-precision settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SchedulerName = Literal["cosine", "step_decay", "none"]


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    scheduler: SchedulerName = "cosine"
    warmup_epochs: int = 3
    amp: bool = True
    grad_clip: float | None = None
    label_smoothing: float = 0.0
