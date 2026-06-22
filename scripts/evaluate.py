"""Evaluate a trained model on a held-out folder dataset.

Computes top-1/5/10 accuracy over the whole set and saves an NxN prediction grid
(predicted character + confidence, green if correct). Evaluates an fp32 checkpoint by
default, or an int8 ``.pte`` via the ExecuTorch runtime with ``--pte``. Defaults to the
``val_root`` of the run config (the dedicated validation set), separate from the
in-training split.

    uv run python scripts/evaluate.py --config configs/runs/kanji_efficientnet_lite_b0.toml
    uv run python scripts/evaluate.py --config configs/runs/kanji_efficientnet_lite_b0.toml \
        --pte outputs/exports/model_xnnpack.pte
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from char_recognition.config import Config, load_config, resolve_device
from char_recognition.data import CharFolderDataset, canonical_class_map, load_labels
from char_recognition.export import load_recognizer
from char_recognition.paths import JP_FONT, resolve_output

Predictor = Callable[[torch.Tensor], torch.Tensor]  # raw image batch -> class probabilities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="run config (for labels + val_root)")
    p.add_argument("--from", dest="ckpt", type=Path, default=None, help="checkpoint (default <out_dir>/best.pt)")
    p.add_argument("--pte", type=Path, default=None, help="evaluate an int8 .pte (ExecuTorch runtime) instead")
    p.add_argument("--root", type=Path, default=None, help="eval dataset root (default data.val_root)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--grid", type=int, default=10, help="render an NxN prediction grid (0 to skip)")
    p.add_argument("--output", type=Path, default=None, help="grid image path (default outputs/eval/predictions.png)")
    return p.parse_args()


def _checkpoint_predictor(
    ckpt: Path, device: torch.device
) -> tuple[Predictor, tuple[int, int], int, int, str]:
    model = load_recognizer(ckpt, map_location=str(device)).to(device).eval()

    def predict(images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model(images.to(device)), dim=1)

    return predict, model.image_size, model.in_channels, model.num_classes, str(ckpt)


def _pte_predictor(pte: Path, cfg: Config) -> tuple[Predictor, tuple[int, int], int, int, str]:
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(str(pte)).load_method("forward")
    image_size, in_channels = cfg.data.image_size, cfg.data.in_channels
    probe = torch.randint(0, 256, (1, in_channels, *image_size)).float()
    num_classes = int(method.execute([probe])[0].shape[1])

    def predict(images: torch.Tensor) -> torch.Tensor:
        # Run the runtime per image: the in-model amax (portable op) only resizes at batch 1,
        # which is also how the .pte runs on device. Slower than batched, but the real artifact.
        outs = [method.execute([images[i : i + 1]])[0] for i in range(images.shape[0])]
        return torch.cat(outs, dim=0)  # the .pte already outputs softmax probabilities

    return predict, image_size, in_channels, num_classes, str(pte)


def _plot_grid(
    n: int,
    images: list[torch.Tensor],
    pred: list[int],
    conf: list[float],
    true: list[int],
    labels: list[str],
    out: Path,
) -> None:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font = fm.FontProperties(fname=str(JP_FONT)) if JP_FONT.exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, n, figsize=(1.4 * n, 1.4 * n))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= len(images):
            continue
        ax.imshow(images[idx][0].numpy(), cmap="gray")
        correct = pred[idx] == true[idx]
        ax.set_title(
            f"{labels[pred[idx]]} {conf[idx]:.2f}",
            fontproperties=font,
            color="green" if correct else "red",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    root = args.root or cfg.data.val_root
    if not root:
        raise SystemExit("no eval root: pass --root or set data.val_root in the config")
    labels = load_labels(cfg.data.labels_file)

    if args.pte:
        if not args.pte.exists():
            raise SystemExit(f".pte not found: {args.pte} (quantize first, or pass an existing --pte)")
        predict, image_size, in_channels, num_classes, source = _pte_predictor(args.pte, cfg)
    else:
        ckpt = args.ckpt or resolve_output(cfg.log.out_dir) / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt} (train first, or pass --from)")
        predict, image_size, in_channels, num_classes, source = _checkpoint_predictor(ckpt, device)

    dataset = CharFolderDataset(
        root, image_size=image_size, in_channels=in_channels,
        class_to_idx=canonical_class_map(root, labels),
    )
    if dataset.num_classes != num_classes:
        raise SystemExit(f"model has {num_classes} classes, eval set has {dataset.num_classes}")
    print(f"eval {source} on {root} | {len(dataset)} images | {dataset.num_classes} classes")

    # Shuffle so the grid samples varied classes; order is irrelevant to accuracy.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    n_grid = args.grid * args.grid
    p_img: list[torch.Tensor] = []
    p_pred: list[int] = []
    p_conf: list[float] = []
    p_true: list[int] = []

    topks = tuple(k for k in (1, 5, 10) if k <= num_classes)
    maxk = max(topks)
    hits = dict.fromkeys(topks, 0)
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval"):
            conf, topk_idx = predict(images).topk(maxk, dim=1)
            conf, topk_idx = conf.cpu(), topk_idx.cpu()
            in_topk = topk_idx == targets.unsqueeze(1)  # (B, maxk): is the true label here?
            for k in topks:
                hits[k] += int(in_topk[:, :k].any(dim=1).sum())
            total += int(targets.numel())
            if len(p_img) < n_grid:
                take = n_grid - len(p_img)
                p_img.extend(images[:take])
                p_pred.extend(topk_idx[:take, 0].tolist())  # top-1 prediction
                p_conf.extend(conf[:take, 0].tolist())
                p_true.extend(targets[:take].tolist())

    for k in topks:
        print(f"top-{k} accuracy: {hits[k] / total:.4f}  ({hits[k]}/{total})")

    if args.grid > 0 and p_img:
        out = args.output or resolve_output("eval") / "predictions.png"
        _plot_grid(args.grid, p_img, p_pred, p_conf, p_true, labels, out)
        print(f"wrote prediction grid: {out}")


if __name__ == "__main__":
    main()
