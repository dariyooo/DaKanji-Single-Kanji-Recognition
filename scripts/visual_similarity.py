"""Rank visually similar characters and write neighbours JSONs + results.csv to outputs/similarity/.

uv run python scripts/visual_similarity.py   # uses outputs/runs/head_rank/r128/best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from char_recognition.config.loader import resolve_device
from char_recognition.data.labels import load_labels
from char_recognition.engine.checkpoint import CheckpointManager
from char_recognition.export.loading import load_recognizer
from char_recognition.paths import LABELS_FILE, RUNS_DIR, resolve_output
from char_recognition.visual_similarity.embed import (
    aggregate_fonts,
    blend_embeddings,
    class_weight_embedding,
    extract_image_embeddings,
)
from char_recognition.visual_similarity.pair_evaluation import (
    CategoryResult,
    evaluate_pairs,
    load_lookalike_pairs,
)
from char_recognition.visual_similarity.render import SIMILARITY_FONTS, render_glyphs
from char_recognition.visual_similarity.similarity_search import top_k_similar, write_neighbours_json

DEFAULT_CKPT = RUNS_DIR / "head_rank" / "r128" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="ckpt", type=Path, default=DEFAULT_CKPT, help="fp32 checkpoint")
    p.add_argument("--pairs", type=Path, default=Path("configs/eval/lookalike_pairs.toml"))
    p.add_argument("--out-dir", type=Path, default=None, help="default outputs/similarity/")
    p.add_argument("--k", type=int, default=10, help="neighbours per character")
    p.add_argument(
        "--min-score",
        default="0",
        help="drop neighbours below this cosine score in the JSON. 'auto' calibrates per embedding "
        "from the lookalike pairs (lowest confirmed-lookalike score), trimming forced filler while "
        "keeping every confirmed hit. Pair metrics are unaffected",
    )
    p.add_argument("--embeddings", nargs="+", default=None, help="subset of embeddings (default: all)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--grid", type=int, default=0, help="rows of a query+neighbours glyph grid (0 to skip)")
    return p.parse_args()


def _auto_cutoff(
    labels: list[str],
    pairs: dict[str, list[tuple[str, str]]],
    indices: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    """Lowest score at which this embedding retrieved a confirmed lookalike: keep everything above."""
    index = {c: i for i, c in enumerate(labels)}
    hit_scores = []
    for pair_list in pairs.values():
        for a, b in pair_list:
            for query, target in ((index[a], index[b]), (index[b], index[a])):
                row = indices[query].tolist()
                if target in row:
                    hit_scores.append(float(scores[query][row.index(target)]))
    return min(hit_scores) if hit_scores else 0.0


def _load_labels(ckpt: Path) -> list[str]:
    """Labels from checkpoint meta (aligned with the trained head), with labels.txt as fallback."""
    meta = CheckpointManager.load(ckpt)["meta"]
    return meta.get("labels") or load_labels(LABELS_FILE)


def _print_table(results: dict[str, dict[str, CategoryResult]], categories: list[str], k: int) -> None:
    header = ["embedding", f"hit@{k}", "MRR", *categories]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    ranked = sorted(results.items(), key=lambda item: item[1]["overall"].mrr, reverse=True)
    for name, by_category in ranked:
        overall = by_category["overall"]
        cells = [name, f"{overall.hit_rate:.2f}", f"{overall.mrr:.3f}"]
        cells += [f"{by_category[c].hits}/{by_category[c].total}" for c in categories]
        print("| " + " | ".join(cells) + " |")


def _write_csv(path: Path, results: dict[str, dict[str, CategoryResult]], k: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embedding", "category", "hits", "total", f"hit@{k}", "mrr", "skipped"])
        for name, by_category in results.items():
            for category, r in by_category.items():
                row = [name, category, r.hits, r.total, f"{r.hit_rate:.4f}", f"{r.mrr:.4f}", r.skipped]
                writer.writerow(row)


def _plot_grid(
    out: Path,
    rows: int,
    k: int,
    labels: list[str],
    glyphs: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    queries: list[int],
) -> None:
    """Query glyph + its top-k neighbour glyphs per row (same style as evaluate.py's grid)."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    from char_recognition.paths import JP_FONT

    font = fm.FontProperties(fname=str(JP_FONT)) if JP_FONT.exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = k + 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.1 * cols, 1.3 * rows))
    for r in range(rows):
        query = queries[r]
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            idx = query if c == 0 else int(indices[query, c - 1])
            title = labels[idx] if c == 0 else f"{labels[idx]} {float(scores[query, c - 1]):.2f}"
            ax.imshow(glyphs[idx, 0].numpy(), cmap="gray")
            ax.set_title(title, fontproperties=font, fontsize=8, color="white" if c == 0 else "gray")
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="black")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt} (train first, or pass --from)")
    device = resolve_device(args.device)
    out_dir = args.out_dir or resolve_output("similarity")

    model = load_recognizer(args.ckpt)
    labels = _load_labels(args.ckpt)
    if len(labels) != model.num_classes:
        raise SystemExit(f"{len(labels)} labels but model has {model.num_classes} classes")
    print(f"loaded {args.ckpt} | {model.backbone_name} | {model.num_classes} classes | device {device}")

    # Render + extract per font, then aggregate each image embedding across fonts.
    per_font: dict[str, list[torch.Tensor]] = {}
    masks: list[torch.Tensor] = []
    mincho_glyphs: torch.Tensor | None = None
    for font_name, font_path in SIMILARITY_FONTS:
        glyphs, covered = render_glyphs(labels, font_path, size=model.image_size)
        print(f"font {font_name}: {int(covered.sum())}/{len(labels)} glyphs covered")
        if font_name == "mincho":
            mincho_glyphs = glyphs
        extracted = extract_image_embeddings(model, glyphs, batch_size=args.batch_size, device=device)
        for name, matrix in extracted.items():
            per_font.setdefault(name, []).append(matrix)
        masks.append(covered)

    embeddings: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for name, matrices in per_font.items():
        embeddings[name] = aggregate_fonts(matrices, masks)
    embeddings["class_weight"] = (class_weight_embedding(model), None)
    if "head" in embeddings:  # ensemble: activations + class weights (same dim, disjoint failures)
        blended = blend_embeddings(embeddings["head"][0], embeddings["class_weight"][0])
        embeddings["head_class_weight"] = (blended, None)  # a zero head row falls back to class_weight
    if args.embeddings:
        unknown = set(args.embeddings) - set(embeddings)
        if unknown:
            raise SystemExit(f"unknown embeddings {sorted(unknown)}, available: {sorted(embeddings)}")
        embeddings = {name: embeddings[name] for name in args.embeddings}

    pairs = load_lookalike_pairs(args.pairs)
    results: dict[str, dict[str, CategoryResult]] = {}
    neighbour_data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (embedding, valid) in embeddings.items():
        indices, scores = top_k_similar(embedding, k=args.k, valid=valid)
        neighbour_data[name] = (indices, scores)
        if args.min_score == "auto":
            cutoff = _auto_cutoff(labels, pairs, indices, scores)
        else:
            cutoff = float(args.min_score)
        path = write_neighbours_json(
            out_dir / f"neighbours_{name}.json", labels, indices, scores, valid, min_score=cutoff
        )
        results[name] = evaluate_pairs(embedding, labels, pairs, k=args.k, valid=valid)
        print(f"embedding {name}: wrote {path} (score cutoff {cutoff:.3f})")

    _print_table(results, list(pairs), args.k)
    csv_path = out_dir / "results.csv"
    _write_csv(csv_path, results, args.k)
    print(f"\nwrote {csv_path}")

    if args.grid > 0 and mincho_glyphs is not None:
        best = max(results, key=lambda name: results[name]["overall"].mrr)
        indices, scores = neighbour_data[best]
        label_index = {c: i for i, c in enumerate(labels)}
        pair_chars = [label_index[c] for pair_list in pairs.values() for p in pair_list for c in p]
        queries = list(dict.fromkeys(pair_chars))[: args.grid]
        grid_path = out_dir / f"grid_{best}.png"
        _plot_grid(grid_path, len(queries), args.k, labels, mincho_glyphs, indices, scores, queries)
        print(f"wrote {grid_path} (best embedding: {best})")


if __name__ == "__main__":
    main()
