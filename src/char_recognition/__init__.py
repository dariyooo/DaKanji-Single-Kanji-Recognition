"""Generic single-character image recognition with PyTorch.

The package is language agnostic: train on any folder-structured image dataset by
supplying a list of labels and a root directory. Preprocessing (resize + normalize)
lives *inside* the model so exported artifacts need no preprocessing on-device.
"""

from __future__ import annotations

import logging

# torchao registers KernelPreference (an Enum) as a pytree constant on import, and torch logs a
# deprecation message for that through the logging module (not warnings). Quiet that internal torch
# logger so the message is gone everywhere; this runs before any (lazy) torchao import.
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

__version__ = "2.0.0"

__all__ = ["__version__"]
