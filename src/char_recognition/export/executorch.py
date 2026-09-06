"""ExecuTorch (.pte) export with dynamic height/width.

Optional dependency, imported lazily (its API moves between releases).
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from char_recognition.export.loading import CAPTURE_CHANNELS, deployable_model, example_input
from char_recognition.optimize.pt2e import DEFAULT_MAX_SIDE, dynamic_input_shapes

__all__ = ["export_executorch"]


def export_executorch(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    dynamic: bool = True,
    max_side: int = DEFAULT_MAX_SIDE,
    probabilities: bool = True,
) -> Path:
    """Lower ``model`` to an ExecuTorch ``.pte`` program. Returns the written path.

    Accepts grayscale (C=1) or RGB (C=3) input; the model reduces colour to one channel.
    """
    import torch
    from torch.export import export

    try:
        from executorch.exir import to_edge
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ExecuTorch is not installed. Install it with `uv sync --extra executorch` "
            "(platform/version sensitive; see the ExecuTorch docs)."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_model = (deployable_model(model) if probabilities else model).eval()
    sample = (example_input(image_size, in_channels=CAPTURE_CHANNELS),)

    # One shared input contract for every dynamic export, so a bound can't drift per backend.
    # Batch is pinned here: this artifact is for single-image inference on device.
    shapes = dynamic_input_shapes(dynamic, max_side=max_side, batch=False)

    with torch.no_grad():
        exported = export(export_model, sample, dynamic_shapes=shapes)
    edge = to_edge(exported)
    program = edge.to_executorch()
    path.write_bytes(program.buffer)
    return path
