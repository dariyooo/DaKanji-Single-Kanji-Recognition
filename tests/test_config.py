"""Config: a run config loads referenced data/log fragments + an inline recipe."""

from __future__ import annotations

from char_recognition.config import load_config
from char_recognition.paths import CONFIGS_DIR


def test_run_config_composition() -> None:
    cfg = load_config(CONFIGS_DIR / "runs" / "synthetic.toml")
    assert cfg.model.name == "efficientnet_lite_b0"  # inline `model = "..."` shorthand
    assert cfg.data.image_size == (64, 64)  # from the data fragment; TOML list -> tuple
    assert cfg.data.synthetic_classes == 10
    assert cfg.optim.epochs == 8  # from the inline [optim] table
    assert cfg.augment.mix_p == 0.5  # from the inline [augment] table
