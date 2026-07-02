"""Data layer: synthetic dataset and augmentation / mix collate."""

from __future__ import annotations

import torch

from char_recognition.config.loader import Config
from char_recognition.data.augment import build_mix_collate, build_train_transform
from char_recognition.data.augmented_dataset import AugmentedDataset
from char_recognition.data.dataset_utils import build_dataloaders, random_split
from char_recognition.data.synthetic import RandomCharDataset


def test_synthetic_dataset_shape() -> None:
    dataset = RandomCharDataset(20, 5, image_size=(64, 64))
    image, label = dataset[0]
    assert image.shape == (1, 64, 64)
    assert 0 <= label < 5
    assert image.min() >= 0.0 and image.max() <= 255.0


def test_augmentation_and_mix_collate() -> None:
    cfg = Config()
    transform = build_train_transform(cfg.augment)
    assert transform(torch.rand(1, 64, 64) * 255).shape == (1, 64, 64)

    collate = build_mix_collate(num_classes=5, cfg=cfg.augment)
    assert collate is not None
    images, targets = collate([(torch.rand(1, 64, 64) * 255, i % 5) for i in range(8)])
    assert images.shape == (8, 1, 64, 64)
    assert targets.shape in {(8,), (8, 5)}  # int labels, or soft labels when mixing fires


def test_mix_collate_with_spawn_workers() -> None:
    # The collate must pickle for DataLoader workers under the 'spawn' start method
    # (macOS/Windows default), i.e. num_workers > 0 with mixup enabled.
    cfg = Config()
    cfg.data.num_workers = 2
    cfg.data.batch_size = 16
    cfg.augment.mix_p = 0.5
    base = RandomCharDataset(64, 6, image_size=(64, 64))
    train_subset, val_subset = random_split(base, 0.2, 0)
    train_dataset = AugmentedDataset(train_subset, build_train_transform(cfg.augment))
    collate = build_mix_collate(6, cfg.augment)
    train_loader, _ = build_dataloaders(
        train_dataset, val_subset, cfg.data, device=torch.device("cpu"), train_collate=collate
    )
    images, _ = next(iter(train_loader))
    assert images.shape[0] == 16
