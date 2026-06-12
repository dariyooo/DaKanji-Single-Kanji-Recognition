"""Dataset splitting and DataLoader construction.

``tf.data`` knobs map to ``num_workers`` + ``prefetch_factor`` (prefetch),
``persistent_workers`` (warm workers) and ``pin_memory``. Augmentation is applied
only to the train split, so validation sees clean images.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split as _torch_random_split

from char_recognition.config import DataConfig

__all__ = ["AugmentedDataset", "build_dataloaders", "random_split"]

Transform = Callable[[Tensor], Tensor]


class AugmentedDataset(Dataset[tuple[Tensor, int]]):
    """Applies a per-sample transform on top of a base dataset/subset."""

    def __init__(self, base: Dataset, transform: Transform | None) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, target = self.base[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def random_split(dataset: Dataset, val_split: float, seed: int) -> tuple[Subset, Subset]:
    """Deterministically split a dataset into (train, val) subsets."""
    total = len(dataset)  # type: ignore[arg-type]
    val_size = round(total * val_split)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = _torch_random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def build_dataloaders(
    train_dataset: Dataset[tuple[Tensor, int]],
    val_dataset: Dataset[tuple[Tensor, int]],
    cfg: DataConfig,
    *,
    device: torch.device,
    train_collate: Callable[[list], tuple[Tensor, Tensor]] | None = None,
) -> tuple[DataLoader[tuple[Tensor, int]], DataLoader[tuple[Tensor, int]]]:
    """Create train/val loaders. ``train_collate`` enables MixUp/CutMix on train."""
    pin_memory = cfg.pin_memory and device.type == "cuda"
    persistent = cfg.persistent_workers and cfg.num_workers > 0
    prefetch = cfg.prefetch_factor if cfg.num_workers > 0 else None
    # Drop the partial last batch only when a full batch remains (avoids 0 batches
    # on small datasets).
    drop_last = len(train_dataset) >= cfg.batch_size  # type: ignore[arg-type]

    train_loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=train_collate,
    )
    val_loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader
