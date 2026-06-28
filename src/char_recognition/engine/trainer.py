"""Custom train/val loop replacing ``model.fit()``.

AMP autocast (+ GradScaler on CUDA), per-epoch scheduler step, manual checkpointing
and logging. Optimizer/scheduler/criterion are injected by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from char_recognition.config import OptimConfig
from char_recognition.engine.checkpoint import CheckpointManager
from char_recognition.engine.logger import MetricLogger

__all__ = ["TrainHistory", "Trainer"]


@dataclass
class TrainHistory:
    """Per-epoch metric history (parallel lists, like the Keras ``History``)."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)

    def append(self, **metrics: float) -> None:
        for key, value in metrics.items():
            getattr(self, key).append(value)


def _to_indices(targets: Tensor) -> Tensor:
    """Soft ``(B, K)`` targets -> hard class indices; int targets pass through."""
    return targets.argmax(dim=1) if targets.ndim == 2 else targets


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        *,
        device: torch.device,
        optim_cfg: OptimConfig,
        logger: MetricLogger | None = None,
        checkpoint: CheckpointManager | None = None,
        log_every: int = 20,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = optim_cfg
        self.logger = logger
        self.checkpoint = checkpoint
        self.log_every = log_every

        # CUDA -> fp16 + GradScaler (mixed_float16). MPS/CPU -> bf16, scaler is a no-op.
        self.amp_enabled = optim_cfg.amp and device.type in ("cuda", "mps")
        self.amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(device.type, enabled=optim_cfg.amp and device.type == "cuda")
        self._global_step = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
        max_steps: int | None = None,
    ) -> TrainHistory:
        """Train for ``epochs``. ``max_steps`` caps batches per epoch (train and val), e.g.
        a single-batch check on a large dataset without sitting through a full pass."""
        epochs = epochs if epochs is not None else self.cfg.epochs
        history = TrainHistory()
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, epoch, epochs, max_steps)
            val_metrics = self._validate(val_loader, max_steps)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            epoch_metrics = {**train_metrics, **val_metrics, "lr": lr}
            history.append(**epoch_metrics)
            if self.logger is not None:
                self.logger.log_metrics({f"epoch/{k}": v for k, v in epoch_metrics.items()}, epoch)
            if self.checkpoint is not None:
                self.checkpoint.save(
                    self.model,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
        return history

    def _train_epoch(
        self, loader: DataLoader, epoch: int, epochs: int, max_steps: int | None = None
    ) -> dict[str, float]:
        self.model.train()
        running_loss, correct, seen = 0.0, 0, 0
        total = len(loader) if max_steps is None else min(max_steps, len(loader))  # bar honors the cap
        progress = tqdm(loader, desc=f"train {epoch + 1}/{epochs}", leave=False, total=total)
        for step, (images, targets) in enumerate(progress):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch = images.size(0)
            running_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == _to_indices(targets)).sum().item()
            seen += batch
            self._global_step += 1
            if self.logger is not None and self._global_step % self.log_every == 0:
                self.logger.log_metrics(
                    {"step/train_loss": loss.item(), "step/train_acc": correct / seen},
                    self._global_step,
                )
            progress.set_postfix(loss=running_loss / seen, acc=correct / seen)
            if max_steps is not None and step + 1 >= max_steps:
                break

        return {"train_loss": running_loss / seen, "train_acc": correct / seen}

    @torch.no_grad()
    def _validate(self, loader: DataLoader, max_steps: int | None = None) -> dict[str, float]:
        self.model.eval()
        running_loss, correct, seen = 0.0, 0, 0
        total = len(loader) if max_steps is None else min(max_steps, len(loader))  # bar honors the cap
        for step, (images, targets) in enumerate(tqdm(loader, desc="val", leave=False, total=total)):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            batch = images.size(0)
            running_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += batch
            if max_steps is not None and step + 1 >= max_steps:
                break
        return {"val_loss": running_loss / seen, "val_acc": correct / seen}

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        """Return class probabilities for a raw image batch ``(B, C, H, W)``."""
        self.model.eval()
        logits = self.model(images.to(self.device))
        return torch.softmax(logits, dim=1)
