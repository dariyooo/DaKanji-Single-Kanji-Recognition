"""Export a trained checkpoint to deployable artifacts: ONNX + ExecuTorch (fp32 ``.pte``).

Both bake resize + normalize into the graph, so they accept a raw grayscale image of any
size. For the int8 / XNNPACK ``.pte``, use ``scripts/quantize.py``. For the Apple CoreML
``.pte``, use ``scripts/export_coreml.py`` (its delegate miscompiles on this toolchain, so it
runs in a separate pinned environment).

    uv run python scripts/export.py --from outputs/runs/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from char_recognition.export import (
    export_executorch,
    export_onnx,
    export_vulkan,
    load_recognizer,
    onnx_parity,
)
from char_recognition.paths import EXPORTS_DIR, RUNS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="ckpt", type=Path, default=RUNS_DIR / "best.pt", help="checkpoint")
    p.add_argument("--onnx", type=Path, default=EXPORTS_DIR / "model.onnx", help="ONNX output path")
    p.add_argument("--pte", type=Path, default=EXPORTS_DIR / "model.pte", help="ExecuTorch fp32 output path")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--static", action="store_true", help="fix the input shape (default: dynamic batch/H/W)")
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-executorch", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="skip the ONNX vs PyTorch parity check")
    # Vulkan delegate (opt-in; cross-platform GPU, fixed-shape; usually slower than XNNPACK here).
    p.add_argument("--vulkan", type=Path, default=None, help="also write a Vulkan-delegated .pte here")
    p.add_argument(
        "--vulkan-quantize",
        type=int,
        choices=(4, 8),
        default=None,
        help="weight-only int4/int8 on the Linear classifier for --vulkan (default: fp32)",
    )
    return p.parse_args()


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt} (train first, or pass --from)")
    model = load_recognizer(args.ckpt)
    dynamic = not args.static
    print(f"loaded {args.ckpt} | {model.backbone_name} | {model.num_classes} classes | {model.image_size}")

    if not args.skip_onnx:
        onnx_path = export_onnx(
            model, args.onnx, image_size=model.image_size, opset=args.opset, dynamic=dynamic
        )
        print(f"wrote {onnx_path} ({_mb(onnx_path):.1f} MB)")
        if not args.no_verify:
            for channels, side, diff in onnx_parity(model, onnx_path, image_size=model.image_size):
                print(f"  parity {channels}ch @ {side}: max|delta| = {diff:.2e}")

    if not args.skip_executorch:
        try:
            pte_path = export_executorch(model, args.pte, image_size=model.image_size, dynamic=dynamic)
            print(f"wrote {pte_path} ({_mb(pte_path):.1f} MB)")
        except ImportError as exc:
            print(f"skipped ExecuTorch (.pte): {exc}")

    # Opt-in Vulkan delegate (fixed-shape; usually slower than XNNPACK CPU for this model). CoreML
    # is intentionally not here: its delegate miscompiles on this toolchain (torch 2.12 /
    # executorch 1.3), so it lives in scripts/export_coreml.py on a pinned torch 2.7 environment.
    if args.vulkan is not None:
        bits = args.vulkan_quantize
        precision = f"int{bits} weights" if bits else "fp32"
        try:
            out = export_vulkan(model, args.vulkan, image_size=model.image_size, weight_bits=bits)
            print(f"wrote {out} ({_mb(out):.1f} MB, Vulkan delegate, {precision}, fixed input)")
        except Exception as exc:  # backend may be absent / unavailable on this platform
            print(f"skipped Vulkan: {type(exc).__name__}: {str(exc).splitlines()[-1][:100]}")


if __name__ == "__main__":
    main()
