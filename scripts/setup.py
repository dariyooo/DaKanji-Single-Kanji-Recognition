"""One-time project setup: extracts fonts/fonts.zip into fonts/.

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
