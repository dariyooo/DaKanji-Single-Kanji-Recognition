"""Manual checkpointing with ``torch.save`` (replaces Keras ModelCheckpoint).

Stores model/optimizer/scheduler state plus rebuild metadata, tracking the best
validation metric across ``best.pt`` and ``last.pt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from char_recognition.models.recognizer import CharRecognizer

__all__ = ["CheckpointManager", "build_checkpoint_meta"]


def build_checkpoint_meta(model: CharRecognizer, labels: list[str] | None = None) -> dict[str, Any]:
    """Metadata persisted with the weights so the model can be rebuilt for export."""
    return {
        "num_classes": model.num_classes,
        "backbone": model.backbone_name,
        "in_channels": model.in_channels,
        "image_size": list(model.image_size),
        "backbone_kwargs": model.backbone_kwargs,
        "labels": labels,
    }


class CheckpointManager:
    def __init__(
        self,
        ckpt_dir: str | Path,
        *,
        monitor: str = "val_acc",
        mode: str = "max",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        # Model-rebuild metadata (backbone, num_classes, image_size, labels, ...)
        # persisted with every checkpoint so deploy/export can reconstruct the model.
        self.meta = meta or {}
        self.best_value: float = float("-inf") if mode == "max" else float("inf")

    def _is_better(self, value: float) -> bool:
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def save(
        self,
        model: nn.Module,
        *,
        epoch: int,
        metrics: dict[str, float],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write ``last.pt`` and, when the monitored metric improves, ``best.pt``."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)  # re-create if removed mid-run
        payload: dict[str, Any] = {
            "epoch": epoch,
            "metrics": metrics,
            "model_state": model.state_dict(),
            "best_value": self.best_value,
            "meta": self.meta,
            **(extra or {}),
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state"] = scheduler.state_dict()

        last_path = self.ckpt_dir / "last.pt"
        torch.save(payload, last_path)

        value = metrics.get(self.monitor)
        if value is not None and self._is_better(value):
            self.best_value = value
            payload["best_value"] = value
            torch.save(payload, self.ckpt_dir / "best.pt")
        return last_path

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)
