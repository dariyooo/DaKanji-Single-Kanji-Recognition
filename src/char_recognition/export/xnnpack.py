"""Lower a PT2E-quantized model to an XNNPACK ExecuTorch ``.pte`` (int8, CPU/ARM).

The model is exported as given. Wrap it in ``ProbabilityModel`` *before* PT2E capture
if you want probability outputs (a converted graph can't be re-wrapped afterwards).
"""

from __future__ import annotations

from pathlib import Path

from torch import nn
from torch.export import export

from char_recognition.export.loading import CAPTURE_CHANNELS, example_input
from char_recognition.optimize.pt2e import CAPTURE_BATCH, dynamic_input_shapes

__all__ = ["export_xnnpack"]


def export_xnnpack(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    dynamic: bool = True,
) -> Path:
    """Lower a converted (int8) model to an XNNPACK ``.pte``. Returns the written path."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Re-export must mirror the capture: same dynamic spec, same batch-/channel-CAPTURE example.
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS, batch=CAPTURE_BATCH),)
    # NB: don't call .eval() here, it's a converted export graph (torchao warns/errors).
    exported = export(model, sample, dynamic_shapes=dynamic_input_shapes(dynamic))
    program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    path.write_bytes(program.buffer)
    return path
