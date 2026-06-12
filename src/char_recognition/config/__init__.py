"""Typed configuration. Each sub-config lives in its own module; loadable from TOML."""

from __future__ import annotations

from char_recognition.config.augment import AugmentConfig
from char_recognition.config.data import DataConfig
from char_recognition.config.loader import Config, Device, list_configs, load_config, resolve_device
from char_recognition.config.log import LogConfig
from char_recognition.config.model import ModelConfig
from char_recognition.config.optim import OptimConfig, SchedulerName

__all__ = [
    "AugmentConfig",
    "Config",
    "DataConfig",
    "Device",
    "LogConfig",
    "ModelConfig",
    "OptimConfig",
    "SchedulerName",
    "list_configs",
    "load_config",
    "resolve_device",
]
