"""Config: a run config loads a referenced data fragment + an inline recipe (incl. log)."""

from __future__ import annotations

from char_recognition.config.loader import load_config
from char_recognition.paths import CONFIGS_DIR


def test_run_config_composition() -> None:
    cfg = load_config(CONFIGS_DIR / "runs" / "synthetic_efficientnet_lite_b0.toml")
    assert cfg.model.name == "efficientnet_lite_b0"  # inline `model = "..."` shorthand
    assert cfg.data.image_size == (64, 64)  # from the data fragment; TOML list -> tuple
    assert cfg.data.synthetic_classes == 10
    assert cfg.optim.epochs == 8  # from the inline [optim] table
    assert cfg.augment.mix_p == 0.5  # from the inline [augment] table
    assert cfg.log.out_dir == "runs/synthetic"  # inline [log] table (no separate configs/log/)
    assert cfg.log.run_name == "synthetic"
