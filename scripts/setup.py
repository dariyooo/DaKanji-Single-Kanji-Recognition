"""One-time project setup: extracts fonts/fonts.zip into fonts/.

uv run scripts/setup.py
"""

from __future__ import annotations

import zipfile

from char_recognition.paths import FONTS_DIR, FONTS_ZIP


def main() -> None:
    if not FONTS_ZIP.exists():
        raise SystemExit(f"font archive not found: {FONTS_ZIP}")
    with zipfile.ZipFile(FONTS_ZIP) as archive:
        names = archive.namelist()
        archive.extractall(FONTS_DIR)
    print(f"extracted {len(names)} fonts into {FONTS_DIR}:")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
