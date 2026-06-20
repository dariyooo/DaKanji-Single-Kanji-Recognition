"""In-model resize + normalize, baked into the exported graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Preprocess(nn.Module):
    """Reduce to 1 channel, resize to a fixed size, and scale each image to ``[0, 1]`` by its max.

    The channel mean makes the model accept grayscale **or** RGB and reduce it to the single
    channel the backbone expects, so an app can feed a raw colour image with no preprocessing.
    It is a no-op for 1-channel input (mean of one value), so grayscale training/inference is
    unchanged. Per-image (not per-batch) normalization keeps single-image inference identical to
    a training batch, which matters for export parity.
    """

    def __init__(self, size: tuple[int, int] = (64, 64), eps: float = 1e-6) -> None:
        super().__init__()
        self.size = size
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # (B, C, H, W), any range, C in {1, 3}
        x = x.to(torch.float32).mean(dim=1, keepdim=True)  # RGB -> gray; identity for C=1
        amax = x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        x = x / amax
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)

    def extra_repr(self) -> str:
        return f"size={self.size}"
