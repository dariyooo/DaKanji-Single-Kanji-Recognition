"""ONNX export with dynamic batch/height/width.

Resize + normalize are in the model, so one artifact serves any runtime input size.
Uses the torch.export-based (dynamo) exporter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.export import Dim

from char_recognition.export.loading import CAPTURE_CHANNELS, deployable_model, example_input

if TYPE_CHECKING:
    import numpy as np

__all__ = ["export_onnx", "export_onnx_int8", "onnx_parity"]


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


def export_onnx_int8(
    fp32_path: str | Path,
    out_path: str | Path,
    calibration: Iterable[np.ndarray],
) -> Path:
    """Static int8 (QDQ, per-channel) quantization of an fp32 ONNX. Returns the written path.

    ``calibration`` yields raw ``(1, C, H, W)`` float32 image batches; static quantization reads
    them to estimate activation ranges, so feed a few hundred representative samples. Conv and
    Gemm/MatMul are quantized (the whole trunk and head); the in-graph resize and RGB-to-gray
    reduction stay fp32.
    """
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    fp32_path, out_path = Path(fp32_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_name = ort.InferenceSession(str(fp32_path)).get_inputs()[0].name

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter({input_name: np.asarray(x, dtype=np.float32)} for x in calibration)

        def get_next(self) -> dict | None:  # pyright: ignore[reportIncompatibleMethodOverride]
            return next(self._it, None)

    quantize_static(
        str(fp32_path),
        str(out_path),
        _Reader(),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
    )
    return out_path


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
