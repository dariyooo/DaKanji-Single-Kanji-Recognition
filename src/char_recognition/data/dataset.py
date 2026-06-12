"""Generic folder dataset: ``root/<class>/*.{png,jpg,...}`` -> ``(C, H, W)`` float in [0, 255].

Language agnostic: labels + a folder, nothing more. Integer-named class folders sort
numerically (the ETL layout), otherwise lexicographically. Augmentation is applied
separately (see datamodule), keeping this class pure.

The file list is held as **compact arrays** (an int label array + a single packed bytes blob
of relative paths with offsets), not millions of Python tuples. This matters at scale: with
the macOS ``spawn`` start method the DataLoader pickles the dataset into every worker, and
buffers transfer in one memcpy each instead of per-tuple. It is cached to a binary manifest
(``.char_index.v2.npz``) in the dataset root: the first run scans every class folder, later
runs reload the manifest in ~seconds. The cache is keyed on the class-folder names; delete
the manifest to force a rescan after adding or removing images within a class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

__all__ = ["CharFolderDataset"]

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")
_MANIFEST_NAME = ".char_index.v2.npz"

# Compact index: int32 class labels, int64 offsets into a packed bytes blob of relative paths.
_Index = tuple[np.ndarray, np.ndarray, bytes]


def _sorted_class_dirs(root: Path) -> list[Path]:
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no class sub-directories found under {root}")
    if all(d.name.isdigit() for d in dirs):
        return sorted(dirs, key=lambda d: int(d.name))
    return sorted(dirs, key=lambda d: d.name)


def _scan(class_dirs: list[Path], extensions: tuple[str, ...]) -> _Index:
    """Walk every class folder into (labels, offsets, blob) with paths relative to root."""
    labels: list[int] = []
    parts: list[bytes] = []
    offsets: list[int] = [0]
    cursor = 0
    for index, class_dir in enumerate(class_dirs):
        prefix = class_dir.name.encode() + b"/"
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in extensions:
                rel = prefix + path.name.encode()
                parts.append(rel)
                cursor += len(rel)
                offsets.append(cursor)
                labels.append(index)
    return np.asarray(labels, dtype=np.int32), np.asarray(offsets, dtype=np.int64), b"".join(parts)


def _write_manifest(manifest: Path, classes: list[str], index: _Index) -> None:
    """Cache the index (best-effort; a read-only root just means we rescan next time)."""
    labels, offsets, blob = index
    try:
        tmp = manifest.with_name(manifest.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(
                f,
                classes=np.frombuffer("\n".join(classes).encode(), dtype=np.uint8),
                labels=labels,
                offsets=offsets,
                blob=np.frombuffer(blob, dtype=np.uint8),
            )
        tmp.replace(manifest)
    except OSError:
        pass


def _read_manifest(manifest: Path, classes: list[str]) -> _Index | None:
    """Reload the manifest; return None if it's missing or its class set no longer matches."""
    try:
        with np.load(manifest) as data:
            if data["classes"].tobytes().decode().split("\n") != classes:
                return None
            return data["labels"], data["offsets"], data["blob"].tobytes()
    except (OSError, ValueError, KeyError, EOFError):
        return None


class CharFolderDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        root: str | Path,
        *,
        image_size: tuple[int, int] = (64, 64),
        in_channels: int = 1,
        extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
        use_cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self._root_prefix = f"{self.root}/"
        self.image_size = image_size
        self.in_channels = in_channels
        self._pil_mode = "L" if in_channels == 1 else "RGB"

        class_dirs = _sorted_class_dirs(self.root)
        self.classes: list[str] = [d.name for d in class_dirs]
        manifest = self.root / _MANIFEST_NAME

        index = _read_manifest(manifest, self.classes) if use_cache else None
        if index is None:
            index = _scan(class_dirs, extensions)
            if use_cache:
                _write_manifest(manifest, self.classes, index)
        self._labels, self._offsets, self._blob = index
        if len(self._labels) == 0:
            raise FileNotFoundError(f"no images with extensions {extensions} under {self.root}")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        rel = self._blob[self._offsets[index] : self._offsets[index + 1]].decode("utf-8")
        target = int(self._labels[index])
        with Image.open(self._root_prefix + rel) as img:
            converted = img.convert(self._pil_mode)
            tensor = F.pil_to_tensor(converted).float()  # (C, H, W) in [0, 255]
        tensor = F.resize(tensor, list(self.image_size), antialias=True)
        return tensor, target
