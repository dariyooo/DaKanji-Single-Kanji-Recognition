"""Grid search over model x input size (fp32), driven by a config.

Trains each (backbone, image_size) and reports accuracy / latency / size: the tool for
picking an architecture. Quantize the winner separately with `scripts/quantize.py`.
Point `[data].root` at a dataset for meaningful numbers; otherwise it runs on synthetic.

    uv run python scripts/grid_search.py --config configs/runs/grid.toml
"""

from __future__ import annotations

import argparse
import copy
import csv
import tomllib
from pathlib import Path

import torch

from char_recognition.config import load_config, resolve_device
from char_recognition.engine import evaluate_accuracy, setup_mlflow, train_from_config
from char_recognition.optimize import benchmark_model
from char_recognition.paths import resolve_output

CPU = torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/runs/grid.toml"))
    p.add_argument("--output", type=Path, default=None, help="results CSV (default <out_dir>/results.csv)")
    p.add_argument("--runs", type=int, default=20, help="timed forward passes per benchmark")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap train+val batches per epoch (bounds each point; ranking stabilizes early). "
        "Full epochs over the whole set when omitted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("rb") as f:
        grid = tomllib.load(f).get("grid", {})
    base = load_config(args.config)
    device = resolve_device(base.device)
    mlflow = setup_mlflow(base.log.mlflow_experiment) if base.log.mlflow else None

    backbones = grid.get("backbones", [base.model.name])
    sizes = grid.get("image_sizes", [base.data.image_size[0]])

    rows: list[dict] = []
    for backbone in backbones:
        for size in sizes:
            row = {"backbone": backbone, "image_size": size}
            try:
                cfg = copy.deepcopy(base)
                cfg.model.name = backbone
                cfg.data.image_size = (size, size)
                cfg.log.out_dir = f"{base.log.out_dir}/{backbone}_{size}"
                cfg.log.mlflow = False  # the grid logs its own runs below (one per point)
                print(f"\n=== {backbone} @ {size}x{size} ===")
                result = train_from_config(cfg, device, max_steps=args.max_steps)

                example = torch.randint(0, 256, (1, cfg.data.in_channels, size, size)).float()
                accuracy = evaluate_accuracy(result.model, result.val_loader, CPU)
                latency_ms, size_mb, params = benchmark_model(
                    result.model, example, device=CPU, runs=args.runs
                )
                row |= {
                    "accuracy": round(accuracy, 4),
                    "latency_ms": round(latency_ms, 3),
                    "size_mb": round(size_mb, 3),
                    "params": params,
                    "error": "",
                }
                _log_point(mlflow, row)
                print(f"  acc={accuracy:.3f} lat={latency_ms:.2f}ms size={size_mb:.2f}MB")
            except Exception as exc:  # a single point failing must not abort the grid
                row |= {
                    "accuracy": None,
                    "latency_ms": None,
                    "size_mb": None,
                    "params": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR {type(exc).__name__}: {exc}")
            rows.append(row)

    _write_results(rows, args.output or resolve_output(base.log.out_dir) / "results.csv")


def _log_point(mlflow, row: dict) -> None:
    """Log one grid point (params + accuracy/latency/size) to MLflow."""
    if mlflow is None:
        return
    run_name = f"{row['backbone']}_{row['image_size']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: row[k] for k in ("backbone", "image_size", "params")})
        mlflow.log_metrics({k: row[k] for k in ("accuracy", "latency_ms", "size_mb")})


def _write_results(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["backbone", "image_size", "accuracy", "latency_ms", "size_mb", "params", "error"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda r: (r["accuracy"] is not None, r["accuracy"] or 0), reverse=True)
    print(f"\nResults ({len(rows)} configs) -> {output}\n")
    print("| backbone | size | acc | latency(ms) | size(MB) |")
    print("|---|---|---|---|---|")
    for r in ranked:
        acc = "-" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        lat = "-" if r["latency_ms"] is None else f"{r['latency_ms']:.2f}"
        mb = "-" if r["size_mb"] is None else f"{r['size_mb']:.2f}"
        cells = [r["backbone"], r["image_size"], acc, lat, mb]
        print("| " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
