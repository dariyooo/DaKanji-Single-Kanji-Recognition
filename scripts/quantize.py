"""Stage 2: quantize a trained fp32 checkpoint for on-device deployment.

PT2E Quantization-Aware *fine-tuning* (conv + linear, int8), lowered to an XNNPACK
ExecuTorch ``.pte`` that runs on CPU/ARM. The fp32 checkpoint stays the source of truth;
this produces a derived, deployable artifact. The QAT fine-tune runs through the same
``Trainer`` as base training, so it gets the same progress bars, MLflow logging, parallel
data loading and ``--max-steps`` (QAT only needs a short fine-tune, so a cap is recommended).

    # Stage 1: uv run python scripts/train.py --config configs/runs/kanji_efficientnet_lite_b0.toml
    # Stage 2:
    uv run python scripts/quantize.py --config configs/runs/kanji_efficientnet_lite_b0.toml \
        --qat-epochs 8 --lr 1e-5 --max-steps 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from char_recognition.config import load_config
from char_recognition.engine import (
    MetricLogger,
    Trainer,
    build_optimizer,
    build_scheduler,
    prepare_data,
)
from char_recognition.export import example_input, export_xnnpack, load_recognizer
from char_recognition.export.loading import CAPTURE_CHANNELS
from char_recognition.models import ProbabilityModel
from char_recognition.optimize import convert_quantized, prepare_xnnpack
from char_recognition.optimize.pt2e import CAPTURE_BATCH
from char_recognition.paths import EXPORTS_DIR, resolve_output


class _ProbNLLLoss(torch.nn.Module):
    """Cross-entropy for a model whose output is already softmax probabilities (deployable graph).

    The exported model bakes in the softmax, so QAT trains on probabilities: ``nll_loss`` on the
    log of the probabilities is exactly cross-entropy on the underlying logits.
    """

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.nll_loss(torch.log(probs.clamp_min(1e-9)), target)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="run config (dataset + model + image size)")
    p.add_argument(
        "--from", dest="ckpt", type=Path, default=None, help="fp32 checkpoint (default <out_dir>/best.pt)"
    )
    p.add_argument("--output", type=Path, default=EXPORTS_DIR / "model_xnnpack.pte")
    p.add_argument("--qat-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5, help="QAT fine-tune learning rate (keep small)")
    p.add_argument("--device", default="cpu", help="QAT device (fake-quant is most reliable on CPU)")
    p.add_argument(
        "--max-steps", type=int, default=None, help="cap batches per QAT epoch / accuracy eval (recommended)"
    )
    return p.parse_args()


@torch.no_grad()
def _int8_accuracy(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, max_steps: int | None = None
) -> float:
    correct = total = 0
    total_batches = len(loader) if max_steps is None else min(max_steps, len(loader))
    progress = tqdm(loader, desc="int8 eval", leave=False, total=total_batches)
    for step, (images, targets) in enumerate(progress):
        preds = model(images.to(device)).argmax(dim=1)
        correct += int((preds == targets.to(device)).sum().item())
        total += int(targets.numel())
        progress.set_postfix(acc=correct / total if total else 0.0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return correct / total if total else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.augment.mix_p = 0.0  # hard targets for the NLL fine-tune; mixing isn't needed here
    cfg.optim.amp = False  # fake-quant runs in fp32
    cfg.optim.epochs = args.qat_epochs
    cfg.optim.lr = args.lr
    cfg.log.run_name = "qat"
    device = torch.device(args.device)

    # Check the checkpoint up front, before building the (potentially slow) dataset.
    ckpt_path = args.ckpt or resolve_output(cfg.log.out_dir) / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path} (train first, or pass --from)")

    train_loader, val_loader, _labels, num_classes = prepare_data(cfg, device)

    # Rebuild the fp32 model from the checkpoint meta, so backbone/size always match the weights.
    model = load_recognizer(ckpt_path, map_location=args.device).train()
    if model.num_classes != num_classes:
        raise SystemExit(f"checkpoint has {model.num_classes} classes, dataset has {num_classes}")
    image_size, in_channels = model.image_size, model.in_channels
    print(f"loaded fp32 weights from {ckpt_path} | {num_classes} classes | {image_size}")

    # Capture the probability model (softmax baked in) and insert QAT fake-quant. Capture at
    # CAPTURE_CHANNELS (RGB) so the .pte accepts colour; the 1-channel fine-tune data still flows
    # through (the channel dim is dynamic, min 1).
    deployable = ProbabilityModel(model).to(device)
    example = (example_input(image_size, in_channels=CAPTURE_CHANNELS, batch=CAPTURE_BATCH).to(device),)
    prepared = cast(GraphModule, prepare_xnnpack(deployable, example, qat=True, dynamic=True))
    # The exported QAT model is already in the correct training state. Switching it with
    # move_exported_model_to_train/eval (what .train()/.eval() trigger) corrupts its fake-quant
    # ranges, so neutralize the Trainer's mode toggles and leave the model exactly as prepared.
    prepared.train = lambda mode=True: prepared  # type: ignore[method-assign]
    prepared.eval = lambda: prepared  # type: ignore[method-assign]

    # QAT fine-tune through the same Trainer as base training: tqdm bars, MLflow, parallel loading.
    optimizer = build_optimizer(prepared, cfg.optim)
    logger = MetricLogger(cfg.log, hparams=cfg.to_dict()) if cfg.log.mlflow else None
    trainer = Trainer(
        prepared,
        optimizer,
        build_scheduler(optimizer, cfg.optim),
        _ProbNLLLoss(),
        device=device,
        optim_cfg=cfg.optim,
        logger=logger,
        log_every=cfg.log.log_every,
    )
    try:
        trainer.fit(train_loader, val_loader, epochs=args.qat_epochs, max_steps=args.max_steps)
    finally:
        if logger is not None:
            logger.close()

    converted = convert_quantized(prepared)
    print(f"int8 val accuracy: {_int8_accuracy(converted, val_loader, device, args.max_steps):.4f}")

    pte = export_xnnpack(converted, args.output, image_size=image_size)
    print(f"wrote {pte} ({pte.stat().st_size / 1e6:.1f} MB)")

    # Sanity-check the artifact actually loads + runs in the ExecuTorch runtime.
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(str(pte)).load_method("forward")
    assert method is not None
    out = method.execute([example_input(image_size, in_channels=in_channels)])[0]
    print(f"runtime check OK: output {tuple(out.shape)}")


if __name__ == "__main__":
    main()
