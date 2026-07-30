"""One-time project setup: run once after cloning.

Currently extracts the committed font archive (fonts/fonts.zip) into fonts/. The
fonts are used for rendering (visual similarity) and plot labels (evaluate.py grid).
Future one-time setup steps belong here too.

    uv run scripts/setup.py
"""

from __future__ import annotations

from char_recognition.paths import FONTS_DIR
from char_recognition.visual_similarity.render import extract_fonts


def main() -> None:
    names = extract_fonts()
    print(f"extracted {len(names)} fonts into {FONTS_DIR}:")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
