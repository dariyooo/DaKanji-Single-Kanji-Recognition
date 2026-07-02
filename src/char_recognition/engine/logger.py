"""Manual MLflow logging.

MLflow is optional and imported lazily; if it isn't installed, logging is skipped with a
warning. Logging is also best-effort at runtime: if the backend errors mid-run, it disables
itself rather than crashing training. The store (sqlite db + artifacts) lives under ``outputs/``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from char_recognition.config.log import LogConfig

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = ["MetricLogger", "setup_mlflow"]


def _enable_sqlite_wal(db_path: Path) -> None:
    """Put the sqlite store in WAL mode so a running ``mlflow ui`` (reader) and the training
    writer don't collide ("database is locked"). WAL is persistent; busy_timeout lets a writer
    wait for a lock instead of erroring. Best-effort — never block training on this."""
    import sqlite3

    try:
        con = sqlite3.connect(db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout=30000;")
        con.close()
    except sqlite3.Error:
        pass


def setup_mlflow(experiment: str, tracking_uri: str | None = None) -> Any:
    """Configure MLflow under ``outputs/`` and select ``experiment``.

    Returns the mlflow module, or ``None`` if mlflow isn't installed.
    """
    try:
        import mlflow
    except ImportError:
        warnings.warn(
            "MLflow logging requested but mlflow is not installed; skipping.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    from char_recognition.paths import MLRUNS_DIR, OUTPUTS_DIR

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # mlflow's file backend is deprecated; use a sqlite store + artifacts dir.
    if tracking_uri is None:
        _enable_sqlite_wal(OUTPUTS_DIR / "mlflow.db")
    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{OUTPUTS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLRUNS_DIR.as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


class MetricLogger:
    """Logs params, scalar metrics and figures to MLflow for one run.

    Best-effort: if an MLflow call fails (store unavailable, locked, removed mid-run, ...),
    it warns once and disables logging so a logging failure never interrupts training.
    """

    def __init__(self, cfg: LogConfig, hparams: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self._mlflow = setup_mlflow(cfg.mlflow_experiment, cfg.mlflow_tracking_uri) if cfg.mlflow else None
        self._guard(lambda m: self._begin(m, hparams))

    def _begin(self, mlflow: Any, hparams: dict[str, Any] | None) -> None:
        mlflow.start_run(run_name=self.cfg.run_name)
        if hparams:
            mlflow.log_params(_flatten(hparams))

    def _guard(self, call: Callable[[Any], None]) -> None:
        """Run an MLflow call; on any failure warn once and disable logging for the run."""
        if self._mlflow is None:
            return
        try:
            call(self._mlflow)
        except Exception as exc:  # logging must never crash training
            warnings.warn(
                f"MLflow logging failed and is now disabled for this run: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._mlflow = None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._guard(lambda m: m.log_metrics(metrics, step=step))

    def log_figure(self, tag: str, figure: Figure, step: int) -> None:
        self._guard(lambda m: m.log_figure(figure, f"{tag}_{step}.png"))

    def close(self) -> None:
        self._guard(lambda m: m.end_run())


def _flatten(d: dict[str, Any], parent: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{parent}.{key}" if parent else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat
