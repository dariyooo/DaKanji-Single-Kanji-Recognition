"""Generic folder dataset: ``root/<class>/*.{png,jpg,...}`` -> ``(C, H, W)`` float in [0, 255].

Language agnostic: labels + a folder, nothing more. By default the class index of a folder is
its sorted position (numeric for integer-named folders, else lexicographic). That assumes
folder names mean the same class across splits; when they don't (e.g. a dataset that numbers
its train and validation folders differently), pass an explicit ``class_to_idx`` mapping —
``canonical_class_map`` builds one from a split's ``encoding.txt`` + a canonical label list.

The file list is held as **compact arrays** (an int label array + a single packed bytes blob
of relative paths with offsets), not millions of Python tuples — so the macOS ``spawn``
DataLoader pickles buffers (one memcpy each) into every worker instead of per-tuple. It is
cached to a binary manifest (``.char_index.v3.npz``) in the dataset root: the first run scans
every class folder, later runs reload in ~seconds. The cache is keyed on the class-folder
names *and* their assigned indices; delete the manifest to force a rescan.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

__all__ = ["CharFolderDataset", "canonical_class_map"]

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")
_MANIFEST_NAME = ".char_index.v3.npz"

# Compact index: int32 class labels, int64 offsets into a packed bytes blob of relative paths.
_Index = tuple[np.ndarray, np.ndarray, bytes]


def canonical_class_map(root: str | Path, labels: Sequence[str]) -> dict[str, int] | None:
    """Map ``folder name -> canonical class index`` (position in ``labels``) via ``root/encoding.txt``.

    Returns ``None`` if the root has no ``encoding.txt`` (generic datasets fall back to folder
    order). Handles both ``{char: [folder, count]}`` and ``{folder: char}`` encoding formats.
    """
    encoding = Path(root) / "encoding.txt"
    if not encoding.exists():
        return None
    parsed = ast.literal_eval(encoding.read_text(encoding="utf-8"))
    folder_to_char: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(value, (list, tuple)):  # {char: [folder, count]}
            folder_to_char[str(value[0])] = key
        else:  # {folder: char}
            folder_to_char[str(key)] = value
    canonical = {char: index for index, char in enumerate(labels)}
    missing = sorted({ch for ch in folder_to_char.values() if ch not in canonical})
    if missing:
        raise ValueError(f"{len(missing)} chars in {encoding} are not in the labels file (e.g. {missing[:3]})")
    return {folder: canonical[char] for folder, char in folder_to_char.items()}


def _sorted_class_dirs(root: Path) -> list[Path]:
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no class sub-directories found under {root}")
    if all(d.name.isdigit() for d in dirs):
        return sorted(dirs, key=lambda d: int(d.name))
    return sorted(dirs, key=lambda d: d.name)


def _scan(class_dirs: list[Path], extensions: tuple[str, ...], mapping: dict[str, int]) -> _Index:
    """Walk every class folder into (labels, offsets, blob); label = ``mapping[folder name]``."""
    labels: list[int] = []
    parts: list[bytes] = []
    offsets: list[int] = [0]
    cursor = 0
    for class_dir in class_dirs:
        index = mapping[class_dir.name]
        prefix = class_dir.name.encode() + b"/"
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in extensions:
                rel = prefix + path.name.encode()
                parts.append(rel)
                cursor += len(rel)
                offsets.append(cursor)
                labels.append(index)
    return np.asarray(labels, dtype=np.int32), np.asarray(offsets, dtype=np.int64), b"".join(parts)


def _class_idx(classes: list[str], mapping: dict[str, int]) -> np.ndarray:
    return np.asarray([mapping[name] for name in classes], dtype=np.int32)


def _write_manifest(manifest: Path, classes: list[str], mapping: dict[str, int], index: _Index) -> None:
    """Cache the index (best-effort; a read-only root just means we rescan next time)."""
    labels, offsets, blob = index
    try:
        tmp = manifest.with_name(manifest.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(
                f,
                classes=np.frombuffer("\n".join(classes).encode(), dtype=np.uint8),
                class_idx=_class_idx(classes, mapping),
                labels=labels,
                offsets=offsets,
                blob=np.frombuffer(blob, dtype=np.uint8),
            )
        tmp.replace(manifest)
    except OSError:
        pass


def _read_manifest(manifest: Path, classes: list[str], mapping: dict[str, int]) -> _Index | None:
    """Reload the manifest; return None if the class folders or their indices changed."""
    try:
        with np.load(manifest) as data:
            if data["classes"].tobytes().decode().split("\n") != classes:
                return None
            if not np.array_equal(data["class_idx"], _class_idx(classes, mapping)):
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
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self._root_prefix = f"{self.root}/"
        self.image_size = image_size
        self.in_channels = in_channels
        self._pil_mode = "L" if in_channels == 1 else "RGB"

        class_dirs = _sorted_class_dirs(self.root)
        self.classes: list[str] = [d.name for d in class_dirs]
        # Default: folder's sorted position is its class index. Override with class_to_idx.
        mapping = class_to_idx or {name: i for i, name in enumerate(self.classes)}
        self._num_classes = max(mapping.values()) + 1
        manifest = self.root / _MANIFEST_NAME

        index = _read_manifest(manifest, self.classes, mapping) if use_cache else None
        if index is None:
            index = _scan(class_dirs, extensions, mapping)
            if use_cache:
                _write_manifest(manifest, self.classes, mapping, index)
        self._labels, self._offsets, self._blob = index
        if len(self._labels) == 0:
            raise FileNotFoundError(f"no images with extensions {extensions} under {self.root}")

    @property
    def num_classes(self) -> int:
        return self._num_classes

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
