"""A tiny convolutional baseline for fast sanity checks and quick experiments."""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["TinyCNN", "tiny_cnn"]


def _conv_bn_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class TinyCNN(nn.Module):
    """Three downsampling stages, global average pool, linear classifier."""

    def __init__(self, num_classes: int, in_channels: int = 1, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_act(in_channels, width),
            _conv_bn_act(width, width, stride=2),
            _conv_bn_act(width, width * 2),
            _conv_bn_act(width * 2, width * 2, stride=2),
            _conv_bn_act(width * 2, width * 4),
            _conv_bn_act(width * 4, width * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def tiny_cnn(num_classes: int, in_channels: int = 1, *, dropout: float = 0.2, **_: object) -> TinyCNN:
    return TinyCNN(num_classes, in_channels=in_channels, dropout=dropout)
