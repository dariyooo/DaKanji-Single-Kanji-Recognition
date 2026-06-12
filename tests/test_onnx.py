"""ONNX export with dynamic batch/height/width."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from char_recognition.models import CharRecognizer


def test_onnx_export_dynamic_shapes() -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    from char_recognition.export import export_onnx

    model = CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = export_onnx(model, Path(tmp) / "m.onnx", image_size=(64, 64), dynamic=True)
        session = ort.InferenceSession(str(path))
        assert session.get_inputs()[0].name == "image"
        for shape in [(1, 1, 64, 64), (2, 1, 120, 90)]:
            x = np.random.randint(0, 256, shape).astype("float32")
            probs = session.run(None, {"image": x})[0]
            assert probs.shape == (shape[0], 8)
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_onnx_parity_matches_pytorch() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from char_recognition.export import export_onnx, onnx_parity

    model = CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = export_onnx(model, Path(tmp) / "m.onnx", image_size=(64, 64), dynamic=True)
        # Includes sizes the graph wasn't traced at, exercising the in-graph resize.
        for _h, _w, diff in onnx_parity(model, path, image_size=(64, 64), in_channels=1):
            assert diff < 1e-4
