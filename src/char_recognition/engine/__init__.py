"""Training engine: trainer loop, optim factories, logging, checkpoints."""

from __future__ import annotations

from char_recognition.engine.checkpoint import CheckpointManager, build_checkpoint_meta
from char_recognition.engine.logger import MetricLogger, setup_mlflow
from char_recognition.engine.optim import build_criterion, build_optimizer, build_scheduler
from char_recognition.engine.runner import (
    TrainingResult,
    evaluate_accuracy,
    prepare_data,
    train_from_config,
)
from char_recognition.engine.trainer import Trainer, TrainHistory

__all__ = [
    "CheckpointManager",
    "MetricLogger",
    "TrainHistory",
    "Trainer",
    "TrainingResult",
    "build_checkpoint_meta",
    "build_criterion",
    "build_optimizer",
    "build_scheduler",
    "evaluate_accuracy",
    "prepare_data",
    "setup_mlflow",
    "train_from_config",
]
