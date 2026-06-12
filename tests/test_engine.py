"""Engine: the training loop learns, and train_from_config runs end to end."""

from __future__ import annotations

import tempfile

import torch

from char_recognition.config import Config
from char_recognition.data import (
    AugmentedDataset,
    RandomCharDataset,
    build_dataloaders,
    build_train_transform,
    random_split,
)
from char_recognition.engine import (
    Trainer,
    build_criterion,
    build_optimizer,
    build_scheduler,
    train_from_config,
)
from char_recognition.models import CharRecognizer

DEVICE = torch.device("cpu")


def test_training_loop_learns() -> None:
    cfg = Config()
    cfg.data.image_size = (64, 64)
    cfg.data.batch_size = 32
    cfg.data.num_workers = 0
    cfg.optim.epochs = 12
    cfg.optim.scheduler = "none"
    cfg.optim.lr = 3e-3
    cfg.optim.warmup_epochs = 0
    cfg.augment.mix_p = 0.0
    cfg.model.name = "tiny_cnn"
    num_classes = 6

    base = RandomCharDataset(360, num_classes, image_size=cfg.data.image_size, noise=0.1)
    train_subset, val_subset = random_split(base, 0.2, cfg.data.seed)
    train_dataset = AugmentedDataset(train_subset, build_train_transform(cfg.augment))
    train_loader, val_loader = build_dataloaders(train_dataset, val_subset, cfg.data, device=DEVICE)

    model = CharRecognizer.from_config(num_classes, cfg.model, cfg.data)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim)
    trainer = Trainer(
        model, optimizer, scheduler, build_criterion(cfg.optim), device=DEVICE, optim_cfg=cfg.optim
    )
    history = trainer.fit(train_loader, val_loader)
    assert history.val_acc[-1] > 0.6, f"expected learning, got {history.val_acc}"


def test_train_from_config_writes_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.model.name = "tiny_cnn"
        cfg.data.image_size = (64, 64)
        cfg.data.synthetic_classes = 6
        cfg.data.num_workers = 0
        cfg.optim.epochs = 2
        cfg.log.out_dir = tmp  # absolute -> used as-is
        cfg.log.mlflow = False

        result = train_from_config(cfg, DEVICE)
        assert result.num_classes == 6
        assert (result.run_dir / "best.pt").exists()
        assert len(result.history.val_acc) == 2


def test_logger_failure_never_crashes() -> None:
    """A backend error must not propagate: the logger warns once and disables itself."""
    from char_recognition.config import LogConfig
    from char_recognition.engine.logger import MetricLogger

    class _Boom:
        def __getattr__(self, _name: str):
            def _raise(*_a: object, **_k: object) -> None:
                raise RuntimeError("backend down")

            return _raise

    logger = MetricLogger(LogConfig(mlflow=False))  # no real backend wired up
    logger._mlflow = _Boom()  # simulate a store that fails on every call
    logger.log_metrics({"loss": 1.0}, step=1)  # must not raise
    assert logger._mlflow is None  # disabled after the first failure
    logger.log_metrics({"loss": 2.0}, step=2)  # now a silent no-op
    logger.close()
