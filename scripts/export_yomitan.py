"""Package a neighbours JSON as a Yomitan kanji-bank dictionary (zip). Static data, no inference.

    uv run python scripts/export_yomitan.py   # reads outputs/similarity/neighbours_head_class_weight.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from char_recognition.paths import resolve_output

DEFAULT_NEIGHBOURS = resolve_output("similarity") / "neighbours_head_class_weight.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--neighbours", type=Path, default=DEFAULT_NEIGHBOURS,
        help="neighbours JSON from scripts/visual_similarity.py (cutoff already applied there)",
    )
    p.add_argument(
        "--out", type=Path, default=resolve_output("similarity") / "dakanji_visual_similarity.zip"
    )
    p.add_argument("--title", default="DaKanji Visual Similarity")
    p.add_argument("--revision", default="1.0")
    p.add_argument("--scores", action="store_true", help="append the cosine score to each entry")
    return p.parse_args()


def kanji_bank(neighbours: dict[str, list[dict]], with_scores: bool) -> list[list]:
    """One row per character with neighbours: [char, onyomi, kunyomi, tags, meanings, stats]."""
    bank = []
    for char, similar in neighbours.items():
        if not similar:
            continue  # no real lookalikes -> no entry (better than forced filler)
        meanings = [
            f"{n['char']} ({n['score']:.2f})" if with_scores else n["char"] for n in similar
        ]
        bank.append([char, "", "", "", meanings, {}])
    return bank


def main() -> None:
    args = parse_args()
    if not args.neighbours.exists():
        raise SystemExit(
            f"neighbours JSON not found: {args.neighbours} (run scripts/visual_similarity.py first)"
        )
    neighbours = json.loads(args.neighbours.read_text(encoding="utf-8"))
    bank = kanji_bank(neighbours, args.scores)

    index = {
        "title": args.title,
        "revision": args.revision,
        "format": 3,
        "author": "CaptainDario (DaAppLab)",
        "url": "https://github.com/CaptainDario/DaKanji-Single-Kanji-Recognition",
        "description": (
            "Visually similar characters (lookalikes), computed from the DaKanji "
            "recognition CNN. The similarity is purely visual and independent of "
            "shared components."
        ),
        "attribution": "Character recognition machine learning model by Dariyooo (DaAppLab)",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.json", json.dumps(index, ensure_ascii=False))
        zf.writestr("kanji_bank_1.json", json.dumps(bank, ensure_ascii=False))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} KB, {len(bank)} kanji entries)")


if __name__ == "__main__":
    main()
