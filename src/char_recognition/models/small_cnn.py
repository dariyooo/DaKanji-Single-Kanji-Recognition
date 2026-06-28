"""A compact hand-rolled CNN, the small end of the architecture sweep.

Bigger and deeper than ``tiny_cnn`` (which exists for test speed): a strided stem plus four
downsampling stages, doubling channels each stage. Plain conv / batchnorm / ReLU6 only, which
keeps it friendly to int8 PT2E quantization and to every ExecuTorch backend. With ``width=32``
the trunk is ~1M params; the 6507-class linear head dominates the total, as with the other
backbones.

``head_rank`` optionally factorizes that head into a low-rank bottleneck
``Linear(feat, r) -> Linear(r, classes)``, which shrinks the dominant tensor (see
``svd_init_factorized_head`` to seed it from a trained full head).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["SmallCNN", "small_cnn", "svd_init_factorized_head"]


def _conv_bn_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class SmallCNN(nn.Module):
    """Strided stem, four (refine + downsample) stages, global pool, linear (or low-rank) head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width: int = 32,
        dropout: float = 0.2,
        head_rank: int | None = None,
    ) -> None:
        super().__init__()
        w = width
        self.head_rank = head_rank
        self.features = nn.Sequential(
            _conv_bn_act(in_channels, w, stride=2),  # stem, /2
            _conv_bn_act(w, w),
            _conv_bn_act(w, w * 2, stride=2),  # /4
            _conv_bn_act(w * 2, w * 2),
            _conv_bn_act(w * 2, w * 4, stride=2),  # /8
            _conv_bn_act(w * 4, w * 4),
            _conv_bn_act(w * 4, w * 8, stride=2),  # /16
            _conv_bn_act(w * 8, w * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        feat = w * 8
        self.classifier: nn.Module = (
            nn.Linear(feat, num_classes)
            if head_rank is None
            else nn.Sequential(nn.Linear(feat, head_rank, bias=False), nn.Linear(head_rank, num_classes))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def small_cnn(
    num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, head_rank: int | None = None, **_: object
) -> SmallCNN:
    return SmallCNN(num_classes, in_channels=in_channels, dropout=dropout, head_rank=head_rank)


@torch.no_grad()
def svd_init_factorized_head(full: nn.Module, factorized: nn.Module, rank: int) -> None:
    """Seed a ``Linear(feat, r) -> Linear(r, classes)`` bottleneck with the rank-r SVD of ``full``.

    ``full`` is the trained ``nn.Linear`` head and ``factorized`` the two-layer bottleneck.
    ``full.weight`` is ``W [classes, feat]``; with ``W = U S Vt`` the bottleneck starts as the best
    rank-r approximation: the first layer gets ``Vt[:r]`` and the second ``U[:, :r] * S[:r]``, so
    one short fine-tune polishes the head instead of learning it from scratch.
    """
    assert isinstance(full, nn.Linear)
    assert isinstance(factorized, nn.Sequential)
    down, up = factorized[0], factorized[1]
    assert isinstance(down, nn.Linear) and isinstance(up, nn.Linear)
    u, s, vt = torch.linalg.svd(full.weight.data, full_matrices=False)  # u[classes,k], s[k], vt[k,feat]
    down.weight.data.copy_(vt[:rank])  # [r, feat]
    up.weight.data.copy_(u[:, :rank] * s[:rank])  # [classes, r]
    if full.bias is not None and up.bias is not None:
        up.bias.data.copy_(full.bias.data)
