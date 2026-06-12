"""Backbone selection and scaling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """``name`` picks a registry backbone."""

    name: str = "efficientnet_lite_b0"
    dropout: float = 0.2
