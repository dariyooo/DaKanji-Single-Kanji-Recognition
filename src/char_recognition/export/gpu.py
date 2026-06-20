"""ExecuTorch accelerator-delegate exports: Vulkan (cross-platform GPU) and CoreML (Apple).

Both are **fixed-shape fp16/fp32** (no int8, no dynamic batch/size). For a small CNN they
usually lose to XNNPACK CPU (dispatch overhead dominates) and they partial-delegate (ops they
don't support fall back to the portable CPU executor). XNNPACK (int8) stays the cross-platform
on-device target; CoreML is the Apple-only option, worth it only if it wins latency or power on
a real device. Each captures a fixed ``(1, CAPTURE_CHANNELS, H, W)`` RGB example; the model
reduces colour to one channel internally (see ``Preprocess``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch import nn

from char_recognition.export.loading import CAPTURE_CHANNELS, deployable_model, example_input

if TYPE_CHECKING:
    from torch.export import ExportedProgram

__all__ = ["export_coreml", "export_vulkan"]


def _export_fixed(model: nn.Module, image_size: tuple[int, int], probabilities: bool) -> ExportedProgram:
    import torch
    from torch.export import export

    export_model = (deployable_model(model) if probabilities else model).eval()
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS),)  # fixed-shape RGB
    with torch.no_grad():
        return export(export_model, sample)


def export_vulkan(
    model: nn.Module, path: str | Path, *, image_size: tuple[int, int], probabilities: bool = True
) -> Path:
    """Lower ``model`` to a Vulkan-delegated ExecuTorch ``.pte`` (fixed ``(1, 3, H, W)`` input)."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = _export_fixed(model, image_size, probabilities)
    program = to_edge_transform_and_lower(exported, partitioner=[VulkanPartitioner()]).to_executorch()
    path.write_bytes(program.buffer)
    return path


def export_coreml(
    model: nn.Module, path: str | Path, *, image_size: tuple[int, int], probabilities: bool = True
) -> Path:
    """Lower ``model`` to a CoreML-delegated ExecuTorch ``.pte`` for Apple (ANE / GPU / CPU).

    fp16 — CoreML can't lower this project's int8 graph (coremltools has no int8 cast). This is
    the blessed Apple path that executorch's MPS-deprecation notice points to. Needs
    ``coremltools`` and runs on iOS/macOS only.
    """
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner
    from executorch.exir import to_edge_transform_and_lower

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = _export_fixed(model, image_size, probabilities)
    specs = CoreMLBackend.generate_compile_specs()  # fp16, compute units = CPU + GPU + ANE
    partitioner = CoreMLPartitioner(compile_specs=specs, take_over_mutable_buffer=False)
    program = to_edge_transform_and_lower(exported, partitioner=[partitioner]).to_executorch()
    path.write_bytes(program.buffer)
    return path
