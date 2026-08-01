"""Per-class embeddings from a trained ``small_cnn``: hooked layers (f3/f5/f5_4x4/f7/pool/head)
and the final Linear weight rows (``class_weight``, needs no images)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from char_recognition.models.recognizer import CharRecognizer
from char_recognition.models.small_cnn import SmallCNN

__all__ = [
    "aggregate_fonts",
    "blend_embeddings",
    "class_weight_embedding",
    "extract_image_embeddings",
    "image_embedding_names",
]

_Pool = Callable[[Tensor], Tensor]


def _gap(x: Tensor) -> Tensor:
    return x.mean(dim=(2, 3))


def _grid_4x4(x: Tensor) -> Tensor:
    return F.adaptive_avg_pool2d(x, 4).flatten(1)


def _flatten(x: Tensor) -> Tensor:
    return x.flatten(1)


def _small_cnn(model: CharRecognizer) -> SmallCNN:
    backbone = model.backbone
    if not isinstance(backbone, SmallCNN):
        raise TypeError(
            f"visual similarity embeddings are defined for small_cnn, got {model.backbone_name!r}"
        )
    return backbone


def _hooked_layers(backbone: SmallCNN) -> dict[str, tuple[nn.Module, _Pool]]:
    layers: dict[str, tuple[nn.Module, _Pool]] = {
        "f3": (backbone.features[3], _gap),
        "f5": (backbone.features[5], _gap),
        "f5_4x4": (backbone.features[5], _grid_4x4),
        "f7": (backbone.features[7], _gap),
        "pool": (backbone.pool, _flatten),
    }
    if isinstance(backbone.classifier, nn.Sequential):  # factorized low-rank head
        layers["head"] = (backbone.classifier[0], _flatten)
    return layers


def image_embedding_names(model: CharRecognizer) -> list[str]:
    """The image-based embedding names available for this model."""
    return list(_hooked_layers(_small_cnn(model)))


def class_weight_embedding(model: CharRecognizer) -> Tensor:
    """``(num_classes, D)`` rows of the final Linear layer: the free, font-less embedding."""
    classifier = _small_cnn(model).classifier
    final = classifier[-1] if isinstance(classifier, nn.Sequential) else classifier
    assert isinstance(final, nn.Linear)
    return final.weight.detach().clone().cpu()  # image embeddings land on cpu too, ranking stays uniform


@torch.no_grad()
def extract_image_embeddings(
    model: CharRecognizer,
    images: Tensor,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """Run raw images through the full model, returning ``{name: (N, D) cpu tensor}`` per layer."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    layers = _hooked_layers(_small_cnn(model))

    captured: dict[str, list[Tensor]] = {name: [] for name in layers}

    def _hook(name: str, pool: _Pool) -> Callable[[nn.Module, tuple, Tensor], None]:
        def hook(_module: nn.Module, _inputs: tuple, output: Tensor) -> None:
            captured[name].append(pool(output).float().cpu())

        return hook

    handles = [
        module.register_forward_hook(_hook(name, pool)) for name, (module, pool) in layers.items()
    ]
    try:
        for start in range(0, images.shape[0], batch_size):
            model(images[start : start + batch_size].to(device))
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(parts) for name, parts in captured.items()}


def blend_embeddings(a: Tensor, b: Tensor) -> Tensor:
    """Equal-weight ensemble: normalize, sum, re-normalize. A zero row falls back to the other input."""
    return F.normalize(F.normalize(a, dim=1) + F.normalize(b, dim=1), dim=1)


def aggregate_fonts(embeddings: list[Tensor], covered: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Masked mean of normalized per-font embeddings. Returns ``(matrix, valid)``, valid = any font covers."""
    total = torch.zeros_like(embeddings[0])
    counts = torch.zeros(embeddings[0].shape[0])
    for emb, mask in zip(embeddings, covered, strict=True):
        total += F.normalize(emb, dim=1) * mask.unsqueeze(1)
        counts += mask.float()
    mean = total / counts.clamp_min(1.0).unsqueeze(1)
    return F.normalize(mean, dim=1), counts > 0
