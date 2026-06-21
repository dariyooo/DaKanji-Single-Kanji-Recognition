# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "executorch==0.6.0",
#     "coremltools<9",
#     "timm>=1.0",
#     "numpy",
#     "pillow",
#     "setuptools<81",
# ]
# ///
"""Export a trained checkpoint to a CoreML-delegated ExecuTorch ``.pte`` (Apple: ANE / GPU / CPU).

CoreML is kept OUT of ``scripts/export.py`` on purpose. Its ExecuTorch delegate is broken on the
main project toolchain (torch 2.12 / executorch 1.3): it lowers without error but produces wrong
predictions (verified 0/12 vs the fp32 reference). It is correct on torch 2.7 / executorch 0.6
(verified 8/8, fp32 parity ~1e-6), so this script pins that older toolchain via uv's inline script
metadata (PEP 723) above. uv builds and caches an isolated environment for it automatically, so the
rest of the pipeline stays on the current stack:

    uv run scripts/export_coreml.py --from outputs/runs/best.pt

The resulting ``.pte`` is a standalone artifact. The model and checkpoint are toolchain independent,
so it is loaded here exactly as on the main stack. fp16 weights (coremltools has no int8 cast for
this graph); fixed ``(1, 3, H, W)`` input, since CoreML does not keep dynamic shapes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This script runs in its own pinned env (declared above), where the project is not installed,
# so put the package source on the path before importing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _ensure_flatc_on_path() -> None:
    """executorch 0.6 serializes the ``.pte`` by shelling out to ``flatc``; add its bundled dir."""
    import shutil

    import executorch

    if shutil.which("flatc"):
        return
    for base in list(getattr(executorch, "__path__", [])):  # namespace package: use __path__
        flatc_dir = Path(base) / "data" / "bin"
        if (flatc_dir / "flatc").exists():
            os.environ["PATH"] = f"{flatc_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="ckpt", type=Path, default=ROOT / "outputs/runs/best.pt", help="checkpoint")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/exports/model_coreml.pte", help="output .pte")
    p.add_argument("--no-verify", action="store_true", help="skip loading + running the .pte to check it")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt} (train first, or pass --from)")

    _ensure_flatc_on_path()
    import torch

    from char_recognition.export.gpu import export_coreml
    from char_recognition.export.loading import load_recognizer

    model = load_recognizer(args.ckpt)
    print(f"loaded {args.ckpt} | {model.backbone_name} | {model.num_classes} classes | {model.image_size}")
    print(f"toolchain: torch {torch.__version__} / executorch 0.6 (CoreML delegate, fp16)")

    out = export_coreml(model, args.out, image_size=model.image_size)  # wraps softmax internally
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, CoreML delegate, fp16, fixed input)")

    if not args.no_verify:
        from executorch.runtime import Runtime

        method = Runtime.get().load_program(str(out)).load_method("forward")
        x = torch.ones(1, 3, *model.image_size).contiguous()  # 0.6 runtime requires contiguous input
        y = method.execute([x])[0]
        ok = tuple(y.shape) == (1, model.num_classes) and abs(float(y.sum()) - 1.0) < 1e-2
        print(f"  self-check: output {tuple(y.shape)} sum={float(y.sum()):.4f} -> {'OK' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit("CoreML .pte produced an unexpected output; not writing as valid")


if __name__ == "__main__":
    main()
