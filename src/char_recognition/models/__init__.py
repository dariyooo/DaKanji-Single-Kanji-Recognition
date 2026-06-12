"""Model definitions: registry of plug-and-play backbones + deployable recognizer."""

from __future__ import annotations

from char_recognition.models.recognizer import CharRecognizer, ProbabilityModel
from char_recognition.models.registry import (
    available_backbones,
    build_backbone,
    register_backbone,
)

__all__ = [
    "CharRecognizer",
    "ProbabilityModel",
    "available_backbones",
    "build_backbone",
    "register_backbone",
]
