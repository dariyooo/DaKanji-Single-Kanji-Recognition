"""Cosine top-k search over an ``(N, D)`` class embedding matrix + the neighbours JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["pair_rank", "top_k_similar", "write_neighbours_json"]


def _masked_similarity(embeddings: Tensor, valid: Tensor | None) -> tuple[Tensor, Tensor]:
    """Normalized embeddings + validity mask (all-valid when ``valid`` is None)."""
    normalized = F.normalize(embeddings, dim=1)
    if valid is None:
        valid = torch.ones(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    return normalized, valid.to(embeddings.device)


def top_k_similar(
    embeddings: Tensor,
    *,
    k: int = 10,
    valid: Tensor | None = None,
    chunk: int = 2048,
) -> tuple[Tensor, Tensor]:
    """Top-``k`` cosine neighbours per row, self and invalid rows excluded. Returns ``(indices, scores)``."""
    normalized, valid_mask = _masked_similarity(embeddings, valid)
    n = normalized.shape[0]
    invalid_columns = torch.zeros(n, device=normalized.device)
    invalid_columns[~valid_mask] = float("-inf")

    all_idx, all_scores = [], []
    for start in range(0, n, chunk):
        block = normalized[start : start + chunk]
        sim = block @ normalized.T + invalid_columns  # (B, N)
        rows = torch.arange(start, min(start + chunk, n))
        sim[torch.arange(rows.numel()), rows] = float("-inf")  # exclude self
        scores, idx = sim.topk(k, dim=1)
        all_idx.append(idx.cpu())
        all_scores.append(scores.cpu())
    return torch.cat(all_idx), torch.cat(all_scores)


def pair_rank(embeddings: Tensor, query: int, target: int, valid: Tensor | None = None) -> int:
    """1-based rank of ``target`` in ``query``'s neighbour list (self excluded)."""
    normalized, valid_mask = _masked_similarity(embeddings, valid)
    sim = normalized[query] @ normalized.T
    sim[query] = float("-inf")
    sim[~valid_mask] = float("-inf")
    return int((sim > sim[target]).sum().item()) + 1


def write_neighbours_json(
    path: str | Path,
    labels: Sequence[str],
    indices: Tensor,
    scores: Tensor,
    valid: Tensor | None = None,
    min_score: float = 0.0,
) -> Path:
    """Write ``{char: [{"char", "score"}, ...]}``. Invalid queries get ``[]``, ``min_score`` drops entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_list = [True] * len(labels) if valid is None else valid.tolist()
    payload = {
        labels[i]: (
            [
                {"char": labels[j], "score": round(float(s), 4)}
                for j, s in zip(indices[i].tolist(), scores[i].tolist(), strict=True)
                if s >= min_score
            ]
            if valid_list[i]
            else []
        )
        for i in range(len(labels))
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
