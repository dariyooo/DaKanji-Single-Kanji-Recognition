"""Headless training CLI.

Examples:
    # Train on a folder dataset
    uv run python scripts/train.py --data-root /data/kanji --labels labels.txt \
        --backbone efficientnet_lite_b0 --image-size 64 --epochs 50

    # Quick end-to-end run on synthetic data
    uv run python scripts/train.py --synthetic --classes 20 --epochs 3

Or load a run config directly:
    uv run python scripts/train.py --config configs/runs/kanji.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from char_recognition.config import Config, load_config, resolve_device
from char_recognition.engine import train_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="TOML config; overridden by flags below")
    p.add_argument("--data-root", type=Path, default=None, help="folder dataset root (root/<class>/*.png)")
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true", help="train on random data")
    p.add_argument("--classes", type=int, default=None, help="number of classes (synthetic mode)")
    p.add_argument("--backbone", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    p.add_argument(
        "--max-steps", type=int, default=None, help="cap batches/epoch (e.g. 1 for a single-batch check)"
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    if args.data_root is not None:
        cfg.data.root = str(args.data_root)
    if args.synthetic:
        cfg.data.root = None
    if args.labels is not None:
        cfg.data.labels_file = str(args.labels)
    if args.classes is not None:
        cfg.data.synthetic_classes = args.classes
    if args.backbone is not None:
        cfg.model.name = args.backbone
    if args.image_size is not None:
        cfg.data.image_size = (args.image_size, args.image_size)
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.no_mixup:
        cfg.augment.mix_p = 0.0
    if args.out_dir is not None:
        cfg.log.out_dir = str(args.out_dir)
    if args.no_mlflow:
        cfg.log.mlflow = False
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    device = resolve_device(cfg.device)
    print(f"device: {device} | backbone: {cfg.model.name} | input: {cfg.data.image_size}")

    result = train_from_config(cfg, device, max_steps=args.max_steps)
    print(f"classes: {result.num_classes} | checkpoints in {result.run_dir}")
    print(f"final: train_acc={result.history.train_acc[-1]:.4f} val_acc={result.history.val_acc[-1]:.4f}")


if __name__ == "__main__":
    main()
