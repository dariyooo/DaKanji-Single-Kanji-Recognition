"""Export: ONNX, ExecuTorch (portable + XNNPACK int8 + Vulkan/CoreML), plus model loading."""

from __future__ import annotations

from char_recognition.export.executorch import export_executorch
from char_recognition.export.gpu import export_coreml, export_vulkan
from char_recognition.export.loading import deployable_model, example_input, load_recognizer
from char_recognition.export.onnx import export_onnx, onnx_parity
from char_recognition.export.xnnpack import export_xnnpack

__all__ = [
    "deployable_model",
    "example_input",
    "export_coreml",
    "export_executorch",
    "export_onnx",
    "export_vulkan",
    "export_xnnpack",
    "load_recognizer",
    "onnx_parity",
]
