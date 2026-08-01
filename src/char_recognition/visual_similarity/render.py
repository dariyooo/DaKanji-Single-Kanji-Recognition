"""Render class labels as white-on-black glyph tensors (the training data polarity),
with a per-font coverage mask so missing glyphs never render as tofu."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from char_recognition.paths import FONTS_DIR, FONTS_ZIP, JP_FONT, JP_FONT_GOTHIC, JP_FONT_HANDWRITING

__all__ = ["SIMILARITY_FONTS", "extract_fonts", "render_glyphs"]

# The three styles the embeddings are averaged over: printed serif/sans + handwriting.
SIMILARITY_FONTS: tuple[tuple[str, Path], ...] = (
    ("mincho", JP_FONT),
    ("gothic", JP_FONT_GOTHIC),
    ("handwriting", JP_FONT_HANDWRITING),
)


def extract_fonts(zip_path: Path = FONTS_ZIP, dest: Path = FONTS_DIR) -> list[str]:
    """Extract the committed font archive. Idempotent. Returns the extracted names."""
    if not zip_path.exists():
        raise FileNotFoundError(f"font archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        archive.extractall(dest)
    return names


def _font_coverage(font_path: Path, labels: Sequence[str]) -> Tensor:
    """Boolean mask: which single-character labels have a real glyph in the font."""
    from fontTools.ttLib import TTFont

    with TTFont(font_path, lazy=True) as font:
        cmap = font.getBestCmap() or {}  # None when the font has no unicode cmap
    return torch.tensor([len(c) == 1 and ord(c) in cmap for c in labels], dtype=torch.bool)


def render_glyphs(
    labels: Sequence[str],
    font_path: Path,
    *,
    size: tuple[int, int] = (64, 64),
    fill_ratio: float = 0.8,
) -> tuple[Tensor, Tensor]:
    """Render labels centered on black. Returns ``(images (N, 1, H, W) in [0, 255], covered mask)``."""
    from PIL import Image, ImageDraw, ImageFont

    if not font_path.exists():
        raise FileNotFoundError(f"font not found: {font_path} (run `uv run scripts/setup.py` first)")

    height, width = size
    font = ImageFont.truetype(str(font_path), int(min(size) * fill_ratio))
    covered = _font_coverage(font_path, labels)

    images = torch.zeros(len(labels), 1, height, width)
    for i, char in enumerate(labels):
        if not covered[i]:
            continue
        canvas = Image.new("L", (width, height), color=0)
        # anchor="mm" centers on the glyph's layout box, which is how CJK fonts center glyphs.
        ImageDraw.Draw(canvas).text((width / 2, height / 2), char, fill=255, font=font, anchor="mm")
        images[i, 0] = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(
            height, width
        )
    return images, covered
