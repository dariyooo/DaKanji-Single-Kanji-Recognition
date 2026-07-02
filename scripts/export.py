"""Export a trained checkpoint to deployable artifacts: ONNX + ExecuTorch (fp32 ``.pte``).

Both bake resize + normalize into the graph, so they accept a raw grayscale image of any size.
Pass ``--config`` to also write a static-int8 ONNX, calibrated on that config's val_root. For
the int8 / XNNPACK ``.pte``, use ``scripts/quantize.py``. For the Apple CoreML ``.pte``, use
``scripts/export_coreml.py`` (its delegate miscompiles on this toolchain, so it runs in a
separate pinned environment).

    uv run python scripts/export.py --from outputs/runs/best.pt   # add --config <run> for int8 ONNX
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from char_recognition.export.executorch import export_executorch
from char_recognition.export.gpu import export_vulkan
from char_recognition.export.loading import load_recognizer
from char_recognition.export.onnx import export_onnx, export_onnx_int8, onnx_parity
from char_recognition.paths import EXPORTS_DIR, RUNS_DIR

if TYPE_CHECKING:
    import numpy as np


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
    p.add_argument("--config", type=Path, default=None, help="run config: enables int8 ONNX")
    p.add_argument("--onnx-int8", type=Path, default=EXPORTS_DIR / "model_int8.onnx", help="int8 ONNX output")
    p.add_argument("--calib-samples", type=int, default=200, help="calibration images for int8 ONNX")
    return p.parse_args()


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def _calibration_images(config_path: Path, n: int) -> list[np.ndarray]:
    """Load up to ``n`` raw val images (shuffled) from a run config, shaped ``(1, C, H, W)``."""
    import random

    from char_recognition.config.loader import load_config
    from char_recognition.data.dataset import CharFolderDataset, canonical_class_map
    from char_recognition.data.labels import load_labels

    cfg = load_config(config_path)
    root = cfg.data.val_root
    if not root:
        raise SystemExit(f"{config_path} sets no data.val_root for int8 ONNX calibration")
    labels = load_labels(cfg.data.labels_file)
    ds = CharFolderDataset(
        root, image_size=cfg.data.image_size, in_channels=cfg.data.in_channels,
        class_to_idx=canonical_class_map(root, labels),
    )
    idx = list(range(len(ds)))
    random.seed(0)
    random.shuffle(idx)
    return [ds[idx[i]][0].unsqueeze(0).numpy() for i in range(min(n, len(ds)))]


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
        if args.config is not None:
            calib = _calibration_images(args.config, args.calib_samples)
            int8_path = export_onnx_int8(onnx_path, args.onnx_int8, calib)
            print(f"wrote {int8_path} ({_mb(int8_path):.1f} MB, static int8, {len(calib)} calib images)")

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
