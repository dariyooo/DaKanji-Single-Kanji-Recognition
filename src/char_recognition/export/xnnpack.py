"""Lower a PT2E-quantized model to an XNNPACK ExecuTorch ``.pte`` (int8, CPU/ARM).

The model is exported as given. Wrap it in ``ProbabilityModel`` *before* PT2E capture
if you want probability outputs (a converted graph can't be re-wrapped afterwards).
"""

from __future__ import annotations

from pathlib import Path

from torch import nn
from torch.export import export

from char_recognition.export.loading import CAPTURE_CHANNELS, example_input
from char_recognition.optimize.pt2e import DEFAULT_MAX_SIDE, dynamic_input_shapes

__all__ = ["export_xnnpack"]


def export_xnnpack(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    dynamic: bool = True,
    max_side: int = DEFAULT_MAX_SIDE,
) -> Path:
    """Lower a converted (int8) model to an XNNPACK ``.pte``. Returns the written path.

    ``max_side`` bounds the accepted height/width and, with it, the runtime memory arena.
    """
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deployment re-export: batch pinned to the example's 1 (dim 0 left out of the spec), so the
    # arena is planned for single-image inference instead of the capture-time batch bound.
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS, batch=1),)
    # NB: don't call .eval() here, it's a converted export graph (torchao warns/errors).
    shapes = dynamic_input_shapes(dynamic, max_side=max_side, batch=False)
    exported = export(model, sample, dynamic_shapes=shapes)
    program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    path.write_bytes(program.buffer)
    return path
