"""Quantization-Aware Training via PT2E (XNNPACK int8) + model benchmarking."""

from __future__ import annotations

from char_recognition.optimize.benchmark import benchmark_model
from char_recognition.optimize.pt2e import (
    convert_quantized,
    dynamic_input_shapes,
    prepare_xnnpack,
)

__all__ = [
    "benchmark_model",
    "convert_quantized",
    "dynamic_input_shapes",
    "prepare_xnnpack",
]
