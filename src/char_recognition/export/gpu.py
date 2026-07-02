"""ExecuTorch accelerator-delegate exports: Vulkan (cross-platform GPU) and CoreML (Apple).

Both are **fixed-shape** (no dynamic batch/size). For a small CNN they usually lose to XNNPACK
CPU (dispatch overhead dominates) and they partial-delegate (ops they don't support fall back to
the portable CPU executor). XNNPACK (int8) stays the cross-platform on-device target; CoreML is
the Apple-only option, worth it only if it wins latency or power on a real device. Each captures
a fixed ``(1, CAPTURE_CHANNELS, H, W)`` RGB example; the model reduces colour to one channel
internally (see ``Preprocess``).

Vulkan can additionally quantize the **Linear classifier** weights to symmetric per-channel
int8 or int4 (``weight_bits``), the only op Vulkan quantizes today. For this model the
classifier is ~71% of the parameters, so weight-only int8/int4 roughly halves / quarters that
tensor while the convs stay fp32. CoreML stays fp16 (coremltools has no int8 cast for this graph).
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


def _export_vulkan_weight_quant(
    model: nn.Module, image_size: tuple[int, int], probabilities: bool, weight_bits: int
) -> ExportedProgram:
    """PT2E weight-only (int8/int4, per-channel symmetric) on Linear; re-export the quant graph."""
    import torch
    from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
        VulkanQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    export_model = (deployable_model(model) if probabilities else model).eval()
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS),)  # fixed-shape RGB
    # is_dynamic=False -> weight-only (activations stay fp32, so no activation calibration needed).
    quantizer = VulkanQuantizer().set_global(
        get_symmetric_quantization_config(is_dynamic=False, weight_bits=weight_bits)
    )
    with torch.no_grad():
        prepared = prepare_pt2e(export(export_model, sample).module(), quantizer)
        prepared(sample[0])  # observe weight ranges
        converted = convert_pt2e(prepared)
        return export(converted, sample)


def export_vulkan(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    probabilities: bool = True,
    weight_bits: int | None = None,
) -> Path:
    """Lower ``model`` to a Vulkan-delegated ExecuTorch ``.pte`` (fixed ``(1, 3, H, W)`` input).

    ``weight_bits=None`` keeps fp32 weights. ``weight_bits`` of 4 or 8 quantizes the Linear
    classifier weights to symmetric per-channel int4/int8 (the only op Vulkan quantizes today);
    convs and activations stay fp32.
    """
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower

    if weight_bits not in (None, 4, 8):
        raise ValueError(f"weight_bits must be None, 4, or 8 (got {weight_bits})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = (
        _export_fixed(model, image_size, probabilities)
        if weight_bits is None
        else _export_vulkan_weight_quant(model, image_size, probabilities, weight_bits)
    )
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
