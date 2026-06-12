"""Rebuilding a deployable model from a checkpoint (checkpoint -> load_recognizer)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from char_recognition.engine import CheckpointManager, build_checkpoint_meta
from char_recognition.export import load_recognizer
from char_recognition.models import CharRecognizer


def test_checkpoint_roundtrip_and_deploy() -> None:
    model = CharRecognizer(5, backbone="tiny_cnn", image_size=(64, 64))
    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(tmp, meta=build_checkpoint_meta(model, [f"c{i}" for i in range(5)]))
        manager.save(model, epoch=0, metrics={"val_acc": 0.5})
        loaded = load_recognizer(Path(tmp) / "best.pt")
    assert loaded.num_classes == 5
    assert loaded.backbone_name == "tiny_cnn"
    assert loaded(torch.rand(1, 1, 70, 30) * 255).shape == (1, 5)
