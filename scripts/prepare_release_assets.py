"""Package the exported artifacts into the DaKanji app's release zips.

The da_lang Asset system installs each by GitHub-release name prefix and unzips it flat into
the model dir, so every zip holds one correctly-named file: char_classifier_{backend}.zip ->
char_classifier.pte, char_classifier_onnx.zip -> char_classifier.onnx, and
char_classifier_labels.zip -> char_classifier_labels.txt. Export first, then run this and
upload outputs/release/ to the data release.

    uv run python scripts/prepare_release_assets.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from char_recognition.paths import EXPORTS_DIR, LABELS_FILE, OUTPUTS_DIR

# da_lang's drawingModelBackends, each mapped to the .pte its export script writes.
BACKEND_PTE = {
    "xnnpack": "model_xnnpack.pte",
    "coreml": "model_coreml.pte",
    "vulkan": "model_vulkan_{variant}.pte",
}
MODEL_ARCNAME = "char_classifier.pte"  # the app loads these exact filenames from disk
ONNX_ARCNAME = "char_classifier.onnx"
LABELS_ARCNAME = "char_classifier_labels.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exports-dir", type=Path, default=EXPORTS_DIR, help="dir with the exported files")
    p.add_argument("--labels", type=Path, default=LABELS_FILE, help="labels file (index <-> character)")
    p.add_argument("--onnx", type=Path, default=None, help="ONNX file (default model_int8.onnx)")
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "release", help="output dir for the zips")
    p.add_argument(
        "--backends", nargs="+", default=list(BACKEND_PTE), choices=list(BACKEND_PTE),
        help="backends to package (default: all)",
    )
    p.add_argument("--vulkan-variant", choices=["int8", "int4"], default="int8", help="Vulkan weight bits")
    return p.parse_args()


def _zip_flat(out_zip: Path, src: Path, arcname: str) -> Path:
    """Write a flat-rooted zip storing ``src`` as ``arcname`` (no directory entries)."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname=arcname)
    return out_zip


def main() -> None:
    args = parse_args()
    onnx = args.onnx or args.exports_dir / "model_int8.onnx"

    # (zip name, source file, name inside the zip) for every asset the release needs.
    jobs: list[tuple[str, Path, str]] = [
        (
            f"char_classifier_{b}.zip",
            args.exports_dir / BACKEND_PTE[b].format(variant=args.vulkan_variant),
            MODEL_ARCNAME,
        )
        for b in args.backends
    ]
    jobs.append(("char_classifier_onnx.zip", onnx, ONNX_ARCNAME))
    jobs.append(("char_classifier_labels.zip", args.labels, LABELS_ARCNAME))

    missing = [str(src) for _, src, _ in jobs if not src.exists()]
    if missing:
        raise SystemExit("missing exports (run the export scripts first):\n  " + "\n  ".join(missing))

    for zip_name, src, arcname in jobs:
        out_zip = _zip_flat(args.out / zip_name, src, arcname)
        print(f"wrote {out_zip.name}  <- {src.name}  ({out_zip.stat().st_size / 1e6:.2f} MB)")

    print(f"\n{len(jobs)} assets in {args.out}")
    print("Upload these to the app's GitHub data release (asset names are matched by prefix).")


if __name__ == "__main__":
    main()
