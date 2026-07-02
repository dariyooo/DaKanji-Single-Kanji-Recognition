"""Checkpoint directory and MLflow tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogConfig:
    out_dir: str = "runs"
    run_name: str | None = None
    log_every: int = 20

    mlflow: bool = True
    mlflow_experiment: str = "dakanji-char-recognition"
    mlflow_tracking_uri: str | None = None  # None => local ./mlruns
