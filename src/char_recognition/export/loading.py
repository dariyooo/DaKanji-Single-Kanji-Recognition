"""Rebuild a trained model from a checkpoint for evaluation or export.

Checkpoints carry the ``meta`` dict written by ``CheckpointManager``, so the model
is reconstructed without a config file.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from char_recognition.engine.checkpoint import CheckpointManager
from char_recognition.models.recognizer import CharRecognizer, ProbabilityModel

# Capture exports with a 3-channel (RGB) example: the model's channel-mean reduces any input
# to 1 channel, and a >1 capture stops the exporter from specializing that mean to a no-op.
# The export's channel dim is marked dynamic (min 1), so the artifact still accepts grayscale.
CAPTURE_CHANNELS = 3


def load_recognizer(checkpoint_path: str | Path, *, map_location: str = "cpu") -> CharRecognizer:
    ckpt = CheckpointManager.load(checkpoint_path, map_location=map_location)
    meta = ckpt["meta"]
    model = CharRecognizer(
        meta["num_classes"],
        backbone=meta["backbone"],
        in_channels=meta["in_channels"],
        image_size=tuple(meta["image_size"]),
        backbone_kwargs=meta.get("backbone_kwargs"),  # rebuild head_rank etc.; None for old checkpoints
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def deployable_model(model: nn.Module) -> ProbabilityModel:
    """Wrap a logits model with softmax for on-device probability output."""
    return ProbabilityModel(model).eval()


def example_input(image_size: tuple[int, int], *, in_channels: int = 1, batch: int = 1) -> Tensor:
    """A raw example image batch for tracing/benchmarking."""
    return torch.randint(0, 256, (batch, in_channels, *image_size)).float()
