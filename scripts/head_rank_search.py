"""Search the classifier-head rank of a pretrained small_cnn (low-rank bottleneck head).

The 6507-class head is ~58% of small_cnn's params. This replaces it with a factorized
``Linear(feat, r) -> Linear(r, 6507)`` bottleneck, SVD-initialized from the trained head so each
variant starts as the best rank-r approximation, then fine-tunes the whole model one epoch per r.
It reports accuracy / latency / size like architecture_search; pick the knee, then quantize +
export the winner (its checkpoint stores ``head_rank`` in meta, so load_recognizer rebuilds it).

    uv run python scripts/head_rank_search.py --from outputs/runs/grid/small_cnn_64/best.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any, cast

import torch

from char_recognition.config.loader import load_config, resolve_device
from char_recognition.engine.logger import setup_mlflow
from char_recognition.engine.runner import evaluate_accuracy, train_from_config
from char_recognition.export.loading import load_recognizer
from char_recognition.models.recognizer import CharRecognizer
from char_recognition.models.small_cnn import SmallCNN, svd_init_factorized_head
from char_recognition.optimize.benchmark import benchmark_model
from char_recognition.paths import resolve_output

CPU = torch.device("cpu")
OUT_DIR = "runs/head_rank"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="ckpt", type=Path, required=True, help="pretrained small_cnn checkpoint")
    p.add_argument("--config", type=Path, default=Path("configs/runs/grid.toml"), help="data + augment")
    p.add_argument("--ranks", type=int, nargs="+", default=[64, 96, 128, 192, 256], help="ranks to search")
    p.add_argument("--epochs", type=int, default=1, help="fine-tune epochs per rank")
    p.add_argument("--lr", type=float, default=1e-4, help="fine-tune lr (low: starting from a trained model)")
    p.add_argument("--runs", type=int, default=20, help="timed forward passes per benchmark")
    p.add_argument("--output", type=Path, default=None, help="results CSV (default <OUT_DIR>/results.csv)")
    return p.parse_args()


def _factorized_from_pretrained(pretrained: CharRecognizer, rank: int) -> CharRecognizer:
    """small_cnn with a rank-r bottleneck head: copy the trained trunk, SVD-init the head."""
    model = CharRecognizer(
        pretrained.num_classes,
        backbone="small_cnn",
        in_channels=pretrained.in_channels,
        image_size=pretrained.image_size,
        backbone_kwargs={"head_rank": rank},
    )
    trunk, pre = cast(SmallCNN, model.backbone), cast(SmallCNN, pretrained.backbone)
    trunk.features.load_state_dict(pre.features.state_dict())  # copy the trained conv trunk
    svd_init_factorized_head(pre.classifier, trunk.classifier, rank)  # SVD-init the bottleneck head
    return model


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt} (train small_cnn first, or pass --from)")

    base = load_config(args.config)
    base.optim.epochs = args.epochs
    base.optim.lr = args.lr
    base.optim.scheduler = "none"  # constant low lr for a short fine-tune
    base.optim.warmup_epochs = 0
    base.log.mlflow_experiment = "dakanji-head-rank"
    device = resolve_device(base.device)
    mlflow = (
        setup_mlflow(base.log.mlflow_experiment, base.log.mlflow_tracking_uri) if base.log.mlflow else None
    )

    pretrained = load_recognizer(args.ckpt)
    if pretrained.backbone_name != "small_cnn":
        raise SystemExit(f"--from must be a small_cnn checkpoint (got {pretrained.backbone_name})")
    print(f"pretrained {args.ckpt} | small_cnn | {pretrained.num_classes} classes | {pretrained.image_size}")

    rows: list[dict] = []
    for rank in args.ranks:
        row: dict = {"head_rank": rank}
        try:
            cfg = copy.deepcopy(base)
            cfg.log.out_dir = f"{OUT_DIR}/r{rank}"
            cfg.log.run_name = f"r{rank}"
            print(f"\n=== head_rank r={rank} ===")
            model = _factorized_from_pretrained(pretrained, rank)
            result = train_from_config(cfg, device, model=model)

            example = torch.randint(0, 256, (1, cfg.data.in_channels, *cfg.data.image_size)).float()
            accuracy = evaluate_accuracy(result.model, result.val_loader, device)
            latency_ms, size_mb, params = benchmark_model(result.model, example, device=CPU, runs=args.runs)
            row |= {
                "accuracy": round(accuracy, 4),
                "latency_ms": round(latency_ms, 3),
                "size_mb": round(size_mb, 3),
                "params": params,
                "error": "",
            }
            _log_point(mlflow, result, row)
            print(f"  acc={accuracy:.3f} lat={latency_ms:.2f}ms size={size_mb:.2f}MB ({params / 1e6:.2f}M)")
        except Exception as exc:  # one rank failing must not abort the search
            row |= {"accuracy": None, "latency_ms": None, "size_mb": None, "params": None,
                    "error": f"{type(exc).__name__}: {exc}"}
            print(f"  ERROR {type(exc).__name__}: {exc}")
        rows.append(row)

    output_path = args.output or resolve_output(OUT_DIR) / "results.csv"
    _write_results(rows, output_path)
    _log_summary(mlflow, output_path)


def _log_point(mlflow: Any, result: Any, row: dict) -> None:
    """Append the rank's final metrics to the run the trainer streamed live during fine-tuning."""
    if mlflow is None:
        return
    run = mlflow.last_active_run()  # the fine-tune run just ended by train_from_config
    if run is None:
        return
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics(
            {k: row[k] for k in ("accuracy", "latency_ms", "size_mb", "params")}
        )
        mlflow.set_tag("checkpoint", str(result.run_dir / "best.pt"))


def _log_summary(mlflow: Any, results_path: Path) -> None:
    """Log the ranked results.csv as an artifact under one summary run."""
    if mlflow is None:
        return
    with mlflow.start_run(run_name="head-rank-summary"):
        mlflow.log_artifact(str(results_path))


def _write_results(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["head_rank", "accuracy", "latency_ms", "size_mb", "params", "error"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda r: (r["accuracy"] is not None, r["accuracy"] or 0), reverse=True)
    print(f"\nResults ({len(rows)} ranks) -> {output}\n")
    print("| head_rank | acc | latency(ms) | size(MB) | params(M) |")
    print("|---|---|---|---|---|")
    for r in ranked:
        acc = "-" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        lat = "-" if r["latency_ms"] is None else f"{r['latency_ms']:.2f}"
        mb = "-" if r["size_mb"] is None else f"{r['size_mb']:.2f}"
        mp = "-" if r["params"] is None else f"{r['params'] / 1e6:.2f}"
        print(f"| {r['head_rank']} | {acc} | {lat} | {mb} | {mp} |")


if __name__ == "__main__":
    main()
