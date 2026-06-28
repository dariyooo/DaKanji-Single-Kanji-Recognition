"""ExecuTorch (.pte) export and on-device runtime execution with dynamic shapes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from char_recognition.models.recognizer import CharRecognizer


def test_executorch_export_runtime() -> None:
    pytest.importorskip("executorch")
    from executorch.runtime import Runtime

    from char_recognition.export.executorch import export_executorch
    from char_recognition.models.recognizer import ProbabilityModel

    model = CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)).eval()
    reference = ProbabilityModel(model).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = export_executorch(model, Path(tmp) / "m.pte", image_size=(64, 64), dynamic=True)
        method = Runtime.get().load_program(str(path)).load_method("forward")
        # Different H/W than the export sample proves the dynamic shapes hold on-device.
        for shape in [(1, 1, 64, 64), (1, 1, 120, 90)]:
            x = torch.randint(0, 256, shape).float()
            out = method.execute([x])[0]
            assert out.shape == (shape[0], 8)
            with torch.no_grad():
                assert torch.allclose(out, reference(x), atol=1e-4)
