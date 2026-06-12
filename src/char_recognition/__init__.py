"""Generic single-character image recognition with PyTorch.

The package is language agnostic: train on any folder-structured image dataset by
supplying a list of labels and a root directory. Preprocessing (resize + normalize)
lives *inside* the model so exported artifacts need no preprocessing on-device.
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
