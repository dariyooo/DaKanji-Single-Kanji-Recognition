"""Lower a PT2E-quantized model to an XNNPACK ExecuTorch ``.pte`` (int8, CPU/ARM).

The model is exported as given. Wrap it in ``ProbabilityModel`` *before* PT2E capture
if you want probability outputs (a converted graph can't be re-wrapped afterwards).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.export import export

from char_recognition.export.loading import CAPTURE_CHANNELS, example_input
from char_recognition.optimize.pt2e import DEFAULT_MAX_SIDE, dynamic_input_shapes

__all__ = ["export_xnnpack", "export_xnnpack_methods", "load_quantized", "save_quantized"]


def save_quantized(model: nn.Module, path: str | Path, *, image_size: tuple[int, int], max_side: int) -> Path:
    """Save a converted int8 graph so it can be re-lowered later without redoing QAT.

    ``max_side`` is a ceiling: the saved program re-exports to any bound at or below it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS, batch=1),)
    shapes = dynamic_input_shapes(True, max_side=max_side, batch=False)
    torch.export.save(export(model, sample, dynamic_shapes=shapes), str(path))
    return path


def load_quantized(path: str | Path) -> nn.Module:
    """Load a graph saved by ``save_quantized``, ready to lower with ``export_xnnpack``."""
    return torch.export.load(str(path)).module()


def export_xnnpack_methods(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    bounds: Sequence[int],
) -> Path:
    """Lower one ``.pte`` holding a method per input bound. Returns the written path.

    Every method is named ``forward_{side}`` after the bound it accepts, so the contract is
    explicit at the call site. Weights are shared across methods, and each method's arena is
    allocated only when that method is loaded.
    """
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS, batch=1),)
    methods = {}
    for side in sorted(set(bounds), reverse=True):
        shapes = dynamic_input_shapes(True, max_side=side, batch=False)
        methods[f"forward_{side}"] = export(model, sample, dynamic_shapes=shapes)
    program = to_edge_transform_and_lower(methods, partitioner=[XnnpackPartitioner()]).to_executorch()
    path.write_bytes(program.buffer)
    return path


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
