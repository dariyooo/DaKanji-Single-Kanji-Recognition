"""Model layer: backbones, in-model preprocessing, configurable input size."""

from __future__ import annotations

import pytest
import torch

from char_recognition.models.recognizer import CharRecognizer
from char_recognition.models.registry import available_backbones


@pytest.mark.parametrize("backbone", available_backbones())
def test_backbone_forward_bchw(backbone: str) -> None:
    model = CharRecognizer(7, backbone=backbone, image_size=(64, 64)).eval()
    # Arbitrary (odd) raw size exercises the in-model resize.
    out = model(torch.randint(0, 256, (3, 1, 81, 53)).float())
    assert out.shape == (3, 7)


def test_preprocess_normalizes_and_resizes() -> None:
    model = CharRecognizer(4, backbone="tiny_cnn", image_size=(64, 64))
    pre = model.preprocess(torch.randint(0, 256, (2, 1, 100, 40)).float())
    assert pre.shape == (2, 1, 64, 64)
    assert pre.min() >= 0.0 and pre.max() <= 1.0 + 1e-5


def test_configurable_input_size() -> None:
    for size in (48, 96, 128):
        model = CharRecognizer(5, backbone="efficientnet_lite_b0", image_size=(size, size)).eval()
        assert model(torch.rand(1, 1, size, size) * 255).shape == (1, 5)
