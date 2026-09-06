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

from char_recognition.paths import EXPORTS_DIR, LABELS_FILE, OUTPUTS_DIR, RUNS_DIR

# Accelerator-delegate backends, each mapped to the .pte its export script writes. Not packaged
# by default: XNNPACK CPU runs this model in ~0.16 ms at batch 1, so the GPU delegates only add
# dispatch overhead and partial delegation. Pass --backends to ship one anyway.
BACKEND_PTE = {
    "coreml": "model_coreml.pte",
    "vulkan": "model_vulkan_{variant}.pte",
}
# One XNNPACK artifact holding a method per input bound, each named after the bound it accepts:
# `forward_1024` (20 MiB arena) and `forward_64` (80 KiB). Weights are shared and each arena is
# allocated only when that method is loaded, so a constrained device pays 80 KiB.
XNNPACK_PTE = "model_xnnpack.pte"
XNNPACK_ZIP = "char_classifier_xnnpack.zip"
MODEL_ARCNAME = "char_classifier.pte"  # the app loads these exact filenames from disk
ONNX_ARCNAME = "char_classifier.onnx"
LABELS_ARCNAME = "char_classifier_labels.txt"
WEIGHTS_ARCNAME = "char_classifier.pt"  # fp32 source of truth, so users can export their own variants
# The quantized graph after QAT. Publishing it means a rebuild at a different input bound never
# needs the fine-tune again, which is otherwise unrecoverable once the .pte is lowered.
QUANTIZED_ARCNAME = "char_classifier.pt2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exports-dir", type=Path, default=EXPORTS_DIR, help="dir with the exported files")
    p.add_argument("--labels", type=Path, default=LABELS_FILE, help="labels file (index <-> character)")
    p.add_argument("--onnx", type=Path, default=None, help="ONNX file (default model_int8.onnx)")
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "release", help="output dir for the zips")
    p.add_argument(
        "--backends",
        nargs="*",
        default=[],
        choices=list(BACKEND_PTE),
        help="accelerator backends to also package (default: none, XNNPACK CPU is the target)",
    )
    p.add_argument("--vulkan-variant", choices=["int8", "int4"], default="int8", help="Vulkan weight bits")
    p.add_argument(
        "--weights",
        type=Path,
        default=RUNS_DIR / "head_rank" / "r128" / "best.pt",
        help="fp32 checkpoint to publish so users can export their own variants",
    )
    p.add_argument("--skip-weights", action="store_true", help="don't publish the fp32 checkpoint")
    p.add_argument(
        "--quantized",
        type=Path,
        default=RUNS_DIR / "head_rank" / "r128" / "quantized.pt2",
        help="quantized graph from the QAT run, so it can be re-lowered without fine-tuning again",
    )
    p.add_argument("--skip-quantized", action="store_true", help="don't publish the quantized graph")
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
    jobs.append((XNNPACK_ZIP, args.exports_dir / XNNPACK_PTE, MODEL_ARCNAME))
    jobs.append(("char_classifier_onnx.zip", onnx, ONNX_ARCNAME))
    jobs.append(("char_classifier_labels.zip", args.labels, LABELS_ARCNAME))
    if not args.skip_weights:
        jobs.append(("char_classifier_weights.zip", args.weights, WEIGHTS_ARCNAME))
    if not args.skip_quantized:
        jobs.append(("char_classifier_quantized.zip", args.quantized, QUANTIZED_ARCNAME))

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
