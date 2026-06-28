"""Wire a Config into a training run. Shared by the CLI, grid search and notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from char_recognition.config import Config
from char_recognition.data import (
    AugmentedDataset,
    CharFolderDataset,
    RandomCharDataset,
    build_dataloaders,
    build_mix_collate,
    build_train_transform,
    canonical_class_map,
    load_labels,
    random_split,
)
from char_recognition.engine.checkpoint import CheckpointManager, build_checkpoint_meta
from char_recognition.engine.logger import MetricLogger
from char_recognition.engine.optim import build_criterion, build_optimizer, build_scheduler
from char_recognition.engine.trainer import Trainer, TrainHistory
from char_recognition.models import CharRecognizer
from char_recognition.paths import resolve_output

__all__ = ["TrainingResult", "evaluate_accuracy", "prepare_data", "train_from_config"]


@dataclass
class TrainingResult:
    model: nn.Module  # CharRecognizer, or a QAT-converted quantized variant
    history: TrainHistory
    val_loader: DataLoader
    labels: list[str]
    num_classes: int
    run_dir: Path


def prepare_data(cfg: Config, device: torch.device) -> tuple[DataLoader, DataLoader, list[str], int]:
    """Build train/val loaders + labels from a config (folder data or synthetic)."""
    base: Dataset[tuple[Tensor, int]]
    if cfg.data.root:
        labels = load_labels(cfg.data.labels_file)
        folder = CharFolderDataset(
            cfg.data.root,
            image_size=cfg.data.image_size,
            in_channels=cfg.data.in_channels,
            class_to_idx=canonical_class_map(cfg.data.root, labels),
        )
        num_classes = folder.num_classes
        base = folder
    else:
        num_classes = cfg.data.synthetic_classes
        base = RandomCharDataset(
            40 * num_classes, num_classes, image_size=cfg.data.image_size, in_channels=cfg.data.in_channels
        )
        labels = [f"cls{i}" for i in range(num_classes)]

    train_subset, val_subset = random_split(base, cfg.data.val_split, cfg.data.seed)
    train_dataset = AugmentedDataset(train_subset, build_train_transform(cfg.augment))
    collate = build_mix_collate(num_classes, cfg.augment)
    train_loader, val_loader = build_dataloaders(
        train_dataset, val_subset, cfg.data, device=device, train_collate=collate
    )
    return train_loader, val_loader, labels, num_classes


def train_from_config(
    cfg: Config, device: torch.device, max_steps: int | None = None, model: CharRecognizer | None = None
) -> TrainingResult:
    """Train an fp32 model from a config (logs to MLflow, writes checkpoints).

    ``max_steps`` caps batches per epoch for a quick check (e.g. one batch on a huge dataset).
    ``model`` (optional) fine-tunes a pre-built model instead of building one from ``cfg.model``
    (e.g. a modified checkpoint). Quantization is a separate Stage 2 (see ``scripts/quantize.py``).
    """
    train_loader, val_loader, labels, num_classes = prepare_data(cfg, device)

    if model is None:
        model = CharRecognizer.from_config(num_classes, cfg.model, cfg.data)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim)
    criterion = build_criterion(cfg.optim)

    run_dir = resolve_output(cfg.log.out_dir)
    logger = MetricLogger(cfg.log, hparams=cfg.to_dict())
    checkpoint = CheckpointManager(run_dir, meta=build_checkpoint_meta(model, labels))
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion,
        device=device,
        optim_cfg=cfg.optim,
        logger=logger,
        checkpoint=checkpoint,
        log_every=cfg.log.log_every,
    )
    try:
        history = trainer.fit(train_loader, val_loader, max_steps=max_steps)
    finally:
        logger.close()  # always end the MLflow run, even if training fails
    return TrainingResult(model, history, val_loader, labels, num_classes, run_dir)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy of ``model`` over ``loader``."""
    model.eval().to(device)
    correct = total = 0
    progress = tqdm(loader, desc="eval", leave=False)
    for images, targets in progress:
        preds = model(images.to(device)).argmax(dim=1)
        correct += (preds == targets.to(device)).sum().item()
        total += int(targets.numel())
        progress.set_postfix(acc=correct / total)
    return correct / total
