"""CharRecognizer = in-model preprocessing + a registry backbone (trained and exported)."""

from __future__ import annotations

from torch import Tensor, nn

from char_recognition.config import DataConfig, ModelConfig
from char_recognition.models.preprocessing import Preprocess
from char_recognition.models.registry import build_backbone

__all__ = ["CharRecognizer", "ProbabilityModel"]


class CharRecognizer(nn.Module):
    """Resize/normalize (in-model) followed by a classification backbone.

    Args:
        num_classes: number of output classes.
        backbone: registry name of the backbone (e.g. ``"efficientnet_lite_b0"``).
        in_channels: input channels (1 for grayscale).
        image_size: fixed size the in-model preprocessing resizes to.
        backbone_kwargs: extra kwargs forwarded to the backbone builder.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        backbone: str = "efficientnet_lite_b0",
        in_channels: int = 1,
        image_size: tuple[int, int] = (64, 64),
        backbone_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.image_size = image_size
        self.backbone_kwargs = dict(backbone_kwargs or {})  # persisted in meta so reload rebuilds it

        self.preprocess = Preprocess(image_size)
        self.backbone = build_backbone(
            backbone, num_classes=num_classes, in_channels=in_channels, **self.backbone_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        """Raw image ``(B, C, H, W)`` (any H/W, any range) -> class logits."""
        return self.backbone(self.preprocess(x))

    @classmethod
    def from_config(cls, num_classes: int, model_cfg: ModelConfig, data_cfg: DataConfig) -> CharRecognizer:
        """Build from :class:`~char_recognition.config.ModelConfig`/``DataConfig``."""
        return cls(
            num_classes,
            backbone=model_cfg.name,
            in_channels=data_cfg.in_channels,
            image_size=data_cfg.image_size,
            backbone_kwargs={"dropout": model_cfg.dropout},
        )


class ProbabilityModel(nn.Module):
    """Wraps a logits model with a softmax for deployment (probabilities out)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.softmax(self.model(x))
