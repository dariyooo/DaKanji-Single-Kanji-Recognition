"""The composite ``Config`` and how to build it: TOML run-config loading + device resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from char_recognition.config.augment import AugmentConfig
from char_recognition.config.data import DataConfig
from char_recognition.config.log import LogConfig
from char_recognition.config.model import ModelConfig
from char_recognition.config.optim import OptimConfig
from char_recognition.paths import CONFIGS_DIR

Device = Literal["auto", "cpu", "cuda", "mps"]


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    log: LogConfig = field(default_factory=LogConfig)
    device: Device = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lists_to_tuples(fields: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """TOML has no tuples, so turn the listed array fields into tuples for the dataclasses."""
    return {k: tuple(v) if k in keys else v for k, v in fields.items()}


def load_config(path: str | Path) -> Config:
    """Load a run config from ``configs/runs/<name>.toml``.

    A run config holds the training recipe inline: ``model`` (a backbone name, or a
    ``[model]`` table) plus ``[optim]`` and ``[augment]`` tables. It references the two
    reusable, environment-specific parts by name: ``data`` and ``log`` load
    ``configs/data/<name>.toml`` / ``configs/log/<name>.toml``. Anything omitted uses defaults.
    """
    import tomllib

    path = Path(path)
    configs_root = path.parent.parent  # configs/  (run configs live in configs/runs/)
    with path.open("rb") as f:
        run = tomllib.load(f)

    def referenced(kind: str, builder: type, tuple_keys: tuple[str, ...] = ()) -> Any:
        """Build from ``configs/<kind>/<name>.toml``, named by the run's ``<kind> = "<name>"``."""
        name = run.get(kind)
        if not name:
            return builder()
        with (configs_root / kind / f"{name}.toml").open("rb") as frag_file:
            fields = tomllib.load(frag_file)
        return builder(**_lists_to_tuples(fields, tuple_keys))

    def inline(key: str, builder: type, tuple_keys: tuple[str, ...] = ()) -> Any:
        """Build from an inline ``[key]`` table in the run config (or defaults if absent)."""
        return builder(**_lists_to_tuples(dict(run.get(key, {})), tuple_keys))

    def model() -> ModelConfig:
        spec = run.get("model")
        if isinstance(spec, str):  # shorthand: bare backbone name
            return ModelConfig(name=spec)
        return ModelConfig(**spec) if spec else ModelConfig()

    return Config(
        data=referenced("data", DataConfig, ("image_size",)),
        log=referenced("log", LogConfig),
        model=model(),
        optim=inline("optim", OptimConfig),
        augment=inline("augment", AugmentConfig, ("cutout_scale", "cutout_ratio")),
        device=run.get("device", "auto"),
    )


def list_configs(configs_dir: str | Path = CONFIGS_DIR) -> list[Path]:
    """List the composable run configs under ``configs/runs/``."""
    return sorted((Path(configs_dir) / "runs").glob("*.toml"))


def resolve_device(device: Device) -> torch.device:
    """Resolve ``"auto"`` to the best available accelerator."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
