"""Data: generic folder dataset, synthetic test data, labels, augmentation."""

from __future__ import annotations

from char_recognition.data.augment import build_mix_collate, build_train_transform
from char_recognition.data.datamodule import (
    AugmentedDataset,
    build_dataloaders,
    random_split,
)
from char_recognition.data.dataset import CharFolderDataset, canonical_class_map
from char_recognition.data.labels import Labels, load_labels
from char_recognition.data.synthetic import RandomCharDataset

__all__ = [
    "AugmentedDataset",
    "CharFolderDataset",
    "Labels",
    "RandomCharDataset",
    "build_dataloaders",
    "build_mix_collate",
    "build_train_transform",
    "canonical_class_map",
    "load_labels",
    "random_split",
]
