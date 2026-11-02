"""Build the gold ranking from FooDB nutrients.

Pipeline (see METRICS.md / README.md):
  1. nutrient matrix Y (n foods x T nutrients)           [load_foodb_nutrients]
  2. log1p(Y)
  3. per-nutrient z-score (mean 0, std 1; zero-var -> 0)
  4. cosine similarity between every pair of foods
  5. drop self pairs (diagonal = -inf)
  6. per query, sort candidates by descending cosine     -> gold ranking
  7. gold top-k saved for k in {5,10,20}
  8. graded relevance for nDCG: rank-based gain (default) or cosine min-max

The gold object never touches graph structure; it is the single shared target
for all graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Gold:
    foods: List[str]  # n food names (query == candidate order)
    sim: np.ndarray  # (n, n) cosine, diagonal = -inf (self excluded)
    order: np.ndarray  # (n, n-1) candidate indices, gold-sorted per query
    topk: Dict[
        int, List[List[int]]
    ]  # k -> per-query list of gold top-k candidate indices
    relevance: np.ndarray  # (n, n) graded relevance gain per (query, candidate)
    relevance_mode: str
    rank_gain_cutoff: int

    @property
    def n(self) -> int:
        return len(self.foods)


def _zscore(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd


def _cosine(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    U = X / nrm
    return U @ U.T


def build_gold(
    foods: List[str],
    Y: np.ndarray,
    *,
    log1p: bool = True,
    zscore: bool = True,
    topk: List[int] = (5, 10, 20),
    relevance: str = "rank_based",
    rank_gain_cutoff: int = 20,
) -> Gold:
    """Construct the gold ranking from a raw nutrient matrix ``Y`` (n, T)."""
    n = len(foods)
    if Y.shape[0] != n:
        raise ValueError(f"Y has {Y.shape[0]} rows but {n} foods.")
    X = Y.astype(np.float64).copy()
    if log1p:
        if np.nanmin(X) < -1.0 + 1e-9:
            raise ValueError(
                "log1p requires values > -1; found smaller. Check nutrient units."
            )
        X = np.log1p(X)
    if zscore:
        X = _zscore(X)

    sim = _cosine(X)
    np.fill_diagonal(sim, -np.inf)  # 5. exclude self

    # 6. per-query gold ranking (descending cosine); stable for ties
    order = np.argsort(-np.where(np.isfinite(sim), sim, -np.inf), axis=1, kind="stable")
    order = order[:, : n - 1]  # drop the self column (now last, value -inf)

    # 7. gold top-k sets
    tk: Dict[int, List[List[int]]] = {}
    for k in topk:
        kk = min(int(k), n - 1)
        tk[int(k)] = [order[q, :kk].tolist() for q in range(n)]

    # 8. graded relevance gain (query x candidate)
    rel = _build_relevance(sim, order, relevance, rank_gain_cutoff)
    return Gold(
        foods=list(foods),
        sim=sim,
        order=order,
        topk=tk,
        relevance=rel,
        relevance_mode=relevance,
        rank_gain_cutoff=int(rank_gain_cutoff),
    )


def _build_relevance(
    sim: np.ndarray, order: np.ndarray, mode: str, cutoff: int
) -> np.ndarray:
    n = sim.shape[0]
    rel = np.zeros((n, n), dtype=np.float64)
    if mode == "rank_based":
        # gold rank 1 -> cutoff, rank 2 -> cutoff-1, ..., rank cutoff -> 1, beyond -> 0
        cutoff = int(cutoff)
        for q in range(n):
            for r in range(min(cutoff, order.shape[1])):
                cand = order[q, r]
                rel[q, cand] = cutoff - r
    elif mode == "cosine_minmax":
        # per-query min-max scaling of finite cosine into [0, 1] (handles negatives)
        for q in range(n):
            row = sim[q].copy()
            finite = np.isfinite(row)
            vals = row[finite]
            lo, hi = vals.min(), vals.max()
            span = hi - lo if hi > lo else 1.0
            scaled = np.zeros(n)
            scaled[finite] = (row[finite] - lo) / span
            rel[q] = scaled
            rel[q, q] = 0.0
    else:
        raise ValueError(f"Unknown relevance mode {mode!r} (rank_based|cosine_minmax).")
    return rel
