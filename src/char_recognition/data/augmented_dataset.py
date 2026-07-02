"""Per-sample transform wrapper over a base dataset.

Augmentation is applied only to the train split, so validation sees clean images.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["AugmentedDataset"]

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
