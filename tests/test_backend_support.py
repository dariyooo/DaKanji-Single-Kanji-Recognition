"""Backend support: the models run on XNNPACK via ExecuTorch (int8) and onnxruntime (ONNX)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from char_recognition.models import CharRecognizer, ProbabilityModel


def test_xnnpack_int8_lowers_and_runs() -> None:
    pytest.importorskip("torchao")
    pytest.importorskip("executorch")
    from executorch.runtime import Runtime

    from char_recognition.export import export_xnnpack
    from char_recognition.optimize import convert_quantized, prepare_xnnpack
    from char_recognition.optimize.pt2e import CAPTURE_BATCH

    # Probability model captured up front (softmax in the graph); PTQ for test speed.
    model = ProbabilityModel(CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)))
    example = (torch.randint(0, 256, (CAPTURE_BATCH, 1, 64, 64)).float(),)
    prepared = prepare_xnnpack(model, example, qat=False, dynamic=True)
    prepared(example[0])  # calibrate observers
    converted = convert_quantized(prepared)

    with tempfile.TemporaryDirectory() as tmp:
        pte = export_xnnpack(converted, Path(tmp) / "m.pte", image_size=(64, 64), dynamic=True)
        method = Runtime.get().load_program(str(pte)).load_method("forward")
        # Two different H/W prove the int8 .pte keeps dynamic shapes on-device.
        for shape in [(1, 1, 64, 64), (1, 1, 128, 90)]:
            out = method.execute([torch.randint(0, 256, shape).float()])[0]
            assert out.shape == (shape[0], 8)
            assert abs(out.sum().item() - 1.0) < 1e-3  # softmax probabilities


def test_xnnpack_onnx_provider_runs() -> None:
    """The ONNX export runs on onnxruntime's XNNPACK execution provider.

    The XNNPACK EP ships only in mobile/custom onnxruntime builds (built with --use_xnnpack);
    a desktop pip wheel silently falls back to CPU. We assert the EP actually loaded and skip
    where it's absent; testing CPU under an XNNPACK label would be misleading.
    """
    import numpy as np

    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if "XnnpackExecutionProvider" not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no XNNPACK execution provider")

    from char_recognition.export import export_onnx

    model = CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64))
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(model, Path(tmp) / "m.onnx", image_size=(64, 64), dynamic=True)
        session = ort.InferenceSession(
            str(onnx_path), providers=["XnnpackExecutionProvider", "CPUExecutionProvider"]
        )
        assert "XnnpackExecutionProvider" in session.get_providers()  # not a silent CPU fallback
        name = session.get_inputs()[0].name
        # Two different H/W prove the ONNX graph keeps dynamic shapes on XNNPACK.
        for shape in [(1, 1, 64, 64), (1, 1, 128, 90)]:
            out = session.run(None, {name: np.random.randint(0, 256, shape).astype("float32")})[0]
            assert out.shape == (shape[0], 8)
            assert abs(float(out.sum()) - 1.0) < 1e-3  # softmax probabilities
