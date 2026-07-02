"""Class labels: an ordered list where index ``i`` is the name of class ``i``.

Formats: ``"lines"`` (one per line) or ``"chars"`` (one per character, the legacy
single-line ``labels.txt``). ``"auto"`` picks ``"chars"`` when there are no newlines.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LabelFormat = Literal["auto", "lines", "chars"]


def load_labels(path: str | Path, fmt: LabelFormat = "auto") -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if fmt == "auto":
        fmt = "chars" if "\n" not in text.strip() else "lines"
    if fmt == "chars":
        return list(text.strip())
    return [line for line in (raw.strip() for raw in text.splitlines()) if line]


@dataclass(frozen=True)
class Labels:
    """Immutable ordered label set with index lookup."""

    names: tuple[str, ...]

    @classmethod
    def from_file(cls, path: str | Path, fmt: LabelFormat = "auto") -> Labels:
        return cls(tuple(load_labels(path, fmt)))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> str:
        return self.names[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def index_of(self, name: str) -> int:
        return self.names.index(name)
