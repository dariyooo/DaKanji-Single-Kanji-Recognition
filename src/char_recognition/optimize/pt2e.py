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


def dynamic_input_shapes(enabled: bool = True) -> dict[str, dict[int, Any]] | None:
    """Dynamic batch + H/W for ``forward(x)``: the "any input size" deployment contract.

    Batch is dynamic so QAT fine-tuning (batch > 1) and on-device inference (batch 1) share
    one captured graph. Capture *and* re-export must pass this same spec with a batch-``CAPTURE_BATCH``
    example; mismatching the two trips an export constraint violation.
    """
    if not enabled:
        return None
    return {
        "x": {
            0: Dim("batch", min=1, max=4096),
            2: Dim("height", min=16, max=4096),
            3: Dim("width", min=16, max=4096),
        }
    }


def _xnnpack_quantizer(qat: bool) -> Any:
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    return XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=qat))


def prepare_xnnpack(model: nn.Module, example: tuple, *, qat: bool, dynamic: bool = True) -> nn.Module:
    """Capture ``model`` and insert observers/fake-quant for the XNNPACK int8 scheme."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e

    captured = export(model, example, dynamic_shapes=dynamic_input_shapes(dynamic)).module()
    prepare = prepare_qat_pt2e if qat else prepare_pt2e
    return prepare(captured, _xnnpack_quantizer(qat))


def convert_quantized(prepared: nn.Module) -> nn.Module:
    """Convert an observed / fake-quantized graph into a real int8 graph."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

    return convert_pt2e(prepared)
