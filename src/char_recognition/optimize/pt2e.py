"""PT2E (export-graph) quantization for XNNPACK int8 deployment.

Graph-based, so it quantizes **Conv2d and Linear** (with conv-BN fusion), unlike the
module-swap quantizers, which only touch Linear. Flow:

    prepare_xnnpack(model, example, qat=True)  # capture + insert fake-quant
        -> fine-tune (recovers accuracy)
    convert_quantized(prepared)                # -> real int8 graph

Lowering the converted graph to a ``.pte`` lives in ``char_recognition.export.xnnpack``.
``qat=False`` does post-training quantization (calibrate with a forward pass instead).
"""

from __future__ import annotations

from typing import Any

from torch import nn
from torch.export import Dim, export

# Capture/re-export the graph at batch 2, not 1: a batch-1 example makes the exporter
# 0/1-specialize the batch dim, collapsing it to a constant and breaking dynamic batch.
# The lowered graph still runs at batch 1 (the Dim min) on device.
CAPTURE_BATCH = 2

# ExecuTorch plans its memory arena statically, to the *declared* upper bounds, at load time and
# before it sees an input. Preprocess runs mean/amax/divide at full input resolution before the
# resize to 64x64, so those intermediates are planned at max_side and the arena grows with its
# square: 64 -> 80 KiB, 1024 -> 20 MiB, 4096 with a dynamic batch -> 4.7 GiB, which the
# low-memory killer takes out on a small device. Bounds cost nothing at runtime, so pick the
# smallest that fits the largest input the caller will send.
DEFAULT_MAX_SIDE = 1024


def dynamic_input_shapes(
    enabled: bool = True, *, max_side: int = DEFAULT_MAX_SIDE, batch: bool = True
) -> dict[str, dict[int, Any]] | None:
    """Dynamic channels + H/W (and optionally batch) for ``forward(x)``: the deployment contract.

    ``batch=True`` (capture/QAT) keeps the batch dim dynamic so fine-tuning at batch > 1 and
    inference at batch 1 share one graph, and needs a batch-``CAPTURE_BATCH`` example: a batch-1
    one makes the exporter 0/1-specialize the dim. ``batch=False`` (deployment) omits dim 0 so it
    specializes to the example's batch, which is how batch gets pinned to 1: ``Dim(min=1, max=1)``
    is rejected outright.
    """
    if not enabled:
        return None
    spec: dict[int, Any] = {
        1: Dim("channels", min=1, max=4),
        2: Dim("height", min=1, max=max_side),
        3: Dim("width", min=1, max=max_side),
    }
    if batch:
        spec[0] = Dim("batch", min=1, max=4096)
    return {"x": spec}


def _xnnpack_quantizer(qat: bool) -> Any:
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    return XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=qat, is_per_channel=True))


def prepare_xnnpack(
    model: nn.Module, example: tuple, *, qat: bool, dynamic: bool = True, max_side: int = DEFAULT_MAX_SIDE
) -> nn.Module:
    """Capture ``model`` and insert observers/fake-quant for the XNNPACK int8 scheme.

    ``max_side`` must be at least the bound the artifact is exported with: the capture bound is a
    ceiling, and re-exporting wider raises a constraint violation.
    """
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e

    captured = export(
        model, example, dynamic_shapes=dynamic_input_shapes(dynamic, max_side=max_side)
    ).module()
    prepare = prepare_qat_pt2e if qat else prepare_pt2e
    return prepare(captured, _xnnpack_quantizer(qat))


def convert_quantized(prepared: nn.Module) -> nn.Module:
    """Convert an observed / fake-quantized graph into a real int8 graph."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

    return convert_pt2e(prepared)
