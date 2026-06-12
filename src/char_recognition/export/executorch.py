"""ExecuTorch (.pte) export with dynamic height/width.

Optional dependency, imported lazily (its API moves between releases).
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from char_recognition.export.loading import deployable_model, example_input

__all__ = ["export_executorch"]


def export_executorch(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: tuple[int, int],
    in_channels: int = 1,
    dynamic: bool = True,
    max_side: int = 1024,
    probabilities: bool = True,
) -> Path:
    """Lower ``model`` to an ExecuTorch ``.pte`` program. Returns the written path."""
    import torch
    from torch.export import Dim, export

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
    sample = (example_input(image_size, in_channels=in_channels),)

    dynamic_shapes = None
    if dynamic:
        # Height/width vary at runtime; batch stays 1 for typical on-device inference.
        height = Dim("height", min=16, max=max_side)
        width = Dim("width", min=16, max=max_side)
        dynamic_shapes = {"x": {2: height, 3: width}}

    with torch.no_grad():
        exported = export(export_model, sample, dynamic_shapes=dynamic_shapes)
    edge = to_edge(exported)
    program = edge.to_executorch()
    path.write_bytes(program.buffer)
    return path
