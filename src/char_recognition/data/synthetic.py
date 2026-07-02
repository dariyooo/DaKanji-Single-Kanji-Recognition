"""Synthetic dataset: a fixed random prototype per class plus noise.

The signal is learnable, so tests can confirm the loop actually trains.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["RandomCharDataset"]


class RandomCharDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        *,
        image_size: tuple[int, int] = (64, 64),
        in_channels: int = 1,
        noise: float = 0.3,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channels = in_channels
        self.noise = noise
        self.seed = seed

        proto_gen = torch.Generator().manual_seed(seed)
        self._prototypes = torch.rand(num_classes, in_channels, *image_size, generator=proto_gen) * 255.0
        label_gen = torch.Generator().manual_seed(seed + 1)
        self._labels = torch.randint(0, num_classes, (num_samples,), generator=label_gen)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        label = int(self._labels[index])
        # Deterministic per-sample noise so the dataset is reproducible across workers.
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        noise = torch.rand(self.in_channels, *self.image_size, generator=gen) - 0.5
        image = (self._prototypes[label] + noise * (2.0 * self.noise * 255.0)).clamp_(0.0, 255.0)
        return image, label
