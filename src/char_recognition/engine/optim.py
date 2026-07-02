"""Optimizer, scheduler and loss factories (built explicitly, passed to the trainer).

``step_decay`` reproduces the legacy Keras schedule; ``cosine`` is the default.
"""

from __future__ import annotations

from torch import nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from char_recognition.config.optim import OptimConfig

__all__ = ["build_criterion", "build_optimizer", "build_scheduler"]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_criterion(cfg: OptimConfig) -> nn.Module:
    # CrossEntropyLoss accepts both int class indices (val) and soft probability
    # targets from MixUp/CutMix (train), so no separate soft-target loss is needed.
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


def _legacy_step_decay(epoch: int) -> float:
    """Original schedule: base_lr * (1 - 0.04 * (epoch // 3)), floored at 0."""
    return max(0.0, 1.0 - 0.04 * (epoch // 3))


def build_scheduler(optimizer: optim.Optimizer, cfg: OptimConfig) -> LRScheduler:
    warmup = cfg.warmup_epochs
    if cfg.scheduler == "none":
        main: LRScheduler = LambdaLR(optimizer, lambda _: 1.0)
    elif cfg.scheduler == "step_decay":
        main = LambdaLR(optimizer, _legacy_step_decay)
    elif cfg.scheduler == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs - warmup))
    else:  # pragma: no cover - guarded by Literal type
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")

    if warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        return SequentialLR(optimizer, [warmup_sched, main], milestones=[warmup])
    return main
