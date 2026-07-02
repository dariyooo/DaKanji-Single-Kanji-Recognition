"""Latency + serialized-size benchmark for a model."""

from __future__ import annotations

import io
import time

import torch
from torch import nn

__all__ = ["benchmark_model"]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_model(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    device: torch.device,
    runs: int = 50,
    warmup: int = 10,
) -> tuple[float, float, int]:
    """Return ``(latency_ms, size_mb, num_params)`` for a single forward pass."""
    model.eval().to(device)
    example_input = example_input.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(example_input)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            model(example_input)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    # Serialized state-dict size; reflects quantized weights (a quantized tensor
    # serializes its low-bit data), unlike summing tensor.element_size().
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / 1e6
    num_params = sum(p.numel() for p in model.parameters())
    return (elapsed / runs) * 1000.0, size_mb, num_params
