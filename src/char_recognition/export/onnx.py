"""ONNX export with dynamic batch/height/width.

Resize + normalize are in the model, so one artifact serves any runtime input size.
Uses the torch.export-based (dynamo) exporter.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim

from char_recognition.export.loading import CAPTURE_CHANNELS, deployable_model, example_input

__all__ = ["export_onnx", "onnx_parity"]


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    opset: int = 18,
    dynamic: bool = True,
    probabilities: bool = True,
) -> Path:
    """Export ``model`` to ONNX. Returns the written path.

    The artifact accepts a raw image ``(B, C, H, W)`` with C = 1 (grayscale) or 3 (RGB); the
    model reduces colour to one channel internally. With ``dynamic`` the batch, channel, height
    and width are all flexible at runtime.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_model = (deployable_model(model) if probabilities else model).eval()
    sample = example_input(image_size, in_channels=CAPTURE_CHANNELS)

    # Keys match the model's forward parameter ("x"); ONNX inputs are renamed below.
    dynamic_shapes = (
        {"x": {0: Dim("batch"), 1: Dim("channels", min=1, max=4), 2: Dim("height"), 3: Dim("width")}}
        if dynamic
        else None
    )
    program = torch.onnx.export(
        export_model,
        (sample,),
        input_names=["image"],
        output_names=["probs"],
        opset_version=opset,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        verbose=False,
    )
    if program is None:  # pragma: no cover - dynamo export always returns a program here
        raise RuntimeError("ONNX export produced no program")
    program.save(str(path))
    return path


def onnx_parity(
    model: nn.Module,
    onnx_path: str | Path,
    *,
    image_size: tuple[int, int] = (64, 64),
    sizes: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]:
    """Max ``|ONNX - PyTorch|`` over a few input sizes and channel counts; needs onnxruntime.

    Compares the exported graph (softmax probabilities) against the model, including the
    in-graph resize and RGB-to-gray, at sizes/channels other than the one traced. Returns
    ``(channels, max_side, max_abs_diff)`` per case.
    """
    import numpy as np
    import onnxruntime as ort

    reference = deployable_model(model)  # softmax baked in, matches export_onnx(probabilities=True)
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    cases = sizes or [image_size, (128, 96), (220, 300)]
    report: list[tuple[int, int, float]] = []
    for (height, width), channels in zip(cases, (1, 3, 1), strict=False):
        sample = np.random.randint(0, 256, (1, channels, height, width)).astype("float32")
        onnx_out = session.run(None, {input_name: sample})[0]
        with torch.no_grad():
            torch_out = reference(torch.from_numpy(sample)).numpy()
        report.append((channels, max(height, width), float(np.abs(onnx_out - torch_out).max())))
    return report
