"""Personalized PageRank (random walk with restart) — model-free, no learning.

Two operators:
  - vanilla PPR : restart walk on the row-normalized (undirected) adjacency.
  - typed-normalized PPR : restart walk on the typed-normalized transition
    (equal mass per relation type) — the MAIN method.

Walk recurrence (restart probability ``alpha``):
    Pi_{t+1} = (1-alpha) * P^T Pi_t + alpha * E
with P row-stochastic and E the one-hot personalization (one column per query).
All queries are solved together as a multi-RHS power iteration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from .graph_preprocess import row_normalize, typed_transition, untyped_adjacency
from .load_graphs import MFGraph


def _ppr_multi(
    P: sp.csr_matrix,
    source_global: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Multi-source PPR. Returns Pi of shape (n_nodes, n_sources)."""
    n = P.shape[0]
    Q = len(source_global)
    Pt = P.T.tocsr()
    E = np.zeros((n, Q), dtype=np.float64)
    E[np.asarray(source_global, dtype=np.int64), np.arange(Q)] = 1.0
    Pi = E.copy()
    for _ in range(int(max_iter)):
        nxt = (1.0 - alpha) * (Pt @ Pi) + alpha * E
        delta = np.abs(nxt - Pi).max()
        Pi = nxt
        if delta < tol:
            break
    return Pi


def run_ppr(
    mfg: MFGraph,
    query_global: np.ndarray,
    candidate_global: np.ndarray,
    cfg: dict,
    *,
    typed: bool,
    relations=None,
) -> np.ndarray:
    """Per-query PPR scores over the candidates.

    Returns an array (n_query, n_candidate); the self cell (query == candidate)
    is set to -inf so it is excluded from ranking.
    """
    ppr_cfg = cfg["inference"]["ppr"]
    alpha = float(ppr_cfg["alpha"])
    directed = bool(ppr_cfg.get("directed", False))
    max_iter = int(ppr_cfg["max_iter"])
    tol = float(ppr_cfg["tol"])

    if typed:
        P = typed_transition(mfg, relations=relations, directed=directed)
    else:
        A = untyped_adjacency(mfg, relations=relations, directed=directed)
        P = row_normalize(A)

    Pi = _ppr_multi(P, query_global, alpha=alpha, max_iter=max_iter, tol=tol)  # (n, Q)
    scores = Pi[np.asarray(candidate_global, dtype=np.int64), :].T  # (Q, C)
    # exclude self
    q_to_c = {int(g): j for j, g in enumerate(candidate_global)}
    for i, g in enumerate(query_global):
        j = q_to_c.get(int(g))
        if j is not None:
            scores[i, j] = -np.inf
    return scores


def ppr_scores_from_transition(
    P: sp.csr_matrix,
    query_global: np.ndarray,
    candidate_global: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    self_exclude: bool = True,
) -> np.ndarray:
    """Lower-level entry (used by budget-matched ego inference)."""
    Pi = _ppr_multi(P, query_global, alpha=alpha, max_iter=max_iter, tol=tol)
    scores = Pi[np.asarray(candidate_global, dtype=np.int64), :].T
    if self_exclude:
        q_to_c = {int(g): j for j, g in enumerate(candidate_global)}
        for i, g in enumerate(query_global):
            j = q_to_c.get(int(g))
            if j is not None:
                scores[i, j] = -np.inf
    return scores
