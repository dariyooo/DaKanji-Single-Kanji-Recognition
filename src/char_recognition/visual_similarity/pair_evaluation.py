"""Score an embedding matrix against curated lookalike pairs (hit@k + MRR, both directions)."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

from char_recognition.visual_similarity.similarity_search import pair_rank

__all__ = ["CategoryResult", "evaluate_pairs", "load_lookalike_pairs"]


def load_lookalike_pairs(path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """``{category: [(a, b), ...]}`` from the pair TOML (each category a list of pairs)."""
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)
    return {category: [(a, b) for a, b in pairs] for category, pairs in raw.items()}


@dataclass
class CategoryResult:
    """Directed-pair metrics for one pair category."""

    hits: int = 0  # partner found in top-k
    total: int = 0  # directed pairs evaluated
    reciprocal_rank_sum: float = 0.0
    skipped: int = 0  # directions whose query/target class was invalid (no glyph)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank_sum / self.total if self.total else 0.0


def evaluate_pairs(
    embeddings: Tensor,
    labels: Sequence[str],
    pairs_by_category: dict[str, list[tuple[str, str]]],
    *,
    k: int = 10,
    valid: Tensor | None = None,
) -> dict[str, CategoryResult]:
    """Per-category metrics plus an ``"overall"`` aggregate. Raises if a pair character is unknown."""
    index = {char: i for i, char in enumerate(labels)}
    missing = sorted({c for pairs in pairs_by_category.values() for p in pairs for c in p if c not in index})
    if missing:
        raise ValueError(f"pair characters not in labels: {missing}")

    valid_list = [True] * len(labels) if valid is None else valid.tolist()
    results: dict[str, CategoryResult] = {}
    overall = CategoryResult()
    for category, pairs in pairs_by_category.items():
        result = CategoryResult()
        for a, b in pairs:
            for query, target in ((index[a], index[b]), (index[b], index[a])):
                if not (valid_list[query] and valid_list[target]):
                    result.skipped += 1
                    continue
                rank = pair_rank(embeddings, query, target, valid)
                result.total += 1
                result.hits += int(rank <= k)
                result.reciprocal_rank_sum += 1.0 / rank
        results[category] = result
        overall.hits += result.hits
        overall.total += result.total
        overall.reciprocal_rank_sum += result.reciprocal_rank_sum
        overall.skipped += result.skipped
    results["overall"] = overall
    return results
