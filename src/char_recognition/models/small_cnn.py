"""A compact hand-rolled CNN, the small end of the architecture sweep.

Bigger and deeper than ``tiny_cnn`` (which exists for test speed): a strided stem plus four
downsampling stages, doubling channels each stage. Plain conv / batchnorm / ReLU6 only, which
keeps it friendly to int8 PT2E quantization and to every ExecuTorch backend. With ``width=32``
the trunk is ~1M params; the 6507-class linear head dominates the total, as with the other
backbones.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["SmallCNN", "small_cnn"]


def _conv_bn_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class SmallCNN(nn.Module):
    """Strided stem, four (refine + downsample) stages, global pool, linear classifier."""

    def __init__(self, num_classes: int, in_channels: int = 1, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        w = width
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
        self.classifier = nn.Linear(w * 8, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def small_cnn(num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object) -> SmallCNN:
    return SmallCNN(num_classes, in_channels=in_channels, dropout=dropout)
