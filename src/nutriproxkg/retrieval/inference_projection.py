"""Projection-based ingredient-ingredient similarity (model-free, no learning).

For each relation that touches Ingredient we project the bipartite graph onto
the ingredients and score every (query, candidate) pair with classic
neighbourhood overlap measures:

  - common_neighbors  : |N(i) ∩ N(j)|
  - jaccard           : |N∩| / |N∪|
  - adamic_adar       : Σ_{z∈N∩} 1/log(deg z)
  - resource_allocation : Σ_{z∈N∩} 1/deg z

All scores are restricted to the 149 query/candidate ingredients; degrees for
Adamic-Adar / Resource-Allocation use the GLOBAL entity degree (over all
ingredients).  The headline "projection" ranking combines the configured metric
across the graph's projectable relations by z-mean (equal weight, no learning).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .graph_preprocess import ingredient_incidence, projectable_relations
from .load_graphs import MFGraph


def _zscore_offdiag(M: np.ndarray) -> np.ndarray:
    """Z-score a similarity matrix using its finite off-diagonal entries."""
    m = M.copy().astype(np.float64)
    mask = np.isfinite(m) & ~np.eye(m.shape[0], dtype=bool)
    vals = m[mask]
    if vals.size == 0:
        return np.zeros_like(m)
    mu, sd = vals.mean(), vals.std()
    sd = sd if sd > 1e-12 else 1.0
    out = (m - mu) / sd
    out[~np.isfinite(M)] = 0.0
    return out


def _relation_similarities(
    mfg: MFGraph, et, ing_local_149: np.ndarray
) -> Dict[str, np.ndarray]:
    """4 overlap measures (149 x 149) for one projectable relation."""
    B, ent_deg, _ = ingredient_incidence(mfg, et)
    B149 = B[ing_local_149]  # (149 x n_entity) sparse
    cn = (B149 @ B149.T).toarray().astype(np.float64)  # common neighbours
    deg = np.asarray(
        B149.sum(axis=1)
    ).ravel()  # entities per ingredient (within 149 rows)
    union = deg[:, None] + deg[None, :] - cn
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, cn / union, 0.0)

    # Adamic-Adar / Resource-Allocation weighting by GLOBAL entity degree
    w_aa = np.zeros_like(ent_deg)
    pos = ent_deg >= 2
    w_aa[pos] = 1.0 / np.log(ent_deg[pos])
    w_ra = np.zeros_like(ent_deg)
    pos1 = ent_deg >= 1
    w_ra[pos1] = 1.0 / ent_deg[pos1]
    import scipy.sparse as sp

    aa = (B149 @ sp.diags(w_aa) @ B149.T).toarray().astype(np.float64)
    ra = (B149 @ sp.diags(w_ra) @ B149.T).toarray().astype(np.float64)

    out = {
        "common_neighbors": cn,
        "jaccard": jac,
        "adamic_adar": aa,
        "resource_allocation": ra,
    }
    for m in out.values():
        np.fill_diagonal(m, -np.inf)  # exclude self
    return out


def run_projection(
    mfg: MFGraph, ing_local_149: np.ndarray, cfg: dict
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Return (combined_score 149x149, per-relation-per-metric breakdown).

    combined_score is the z-mean (over relations) of the configured
    ``inference.projection.combine_metric`` and is used as the "projection"
    method ranking.  ``breakdown[rel_name][metric]`` holds every component.
    """
    combine_metric = cfg["inference"]["projection"].get("combine_metric", "adamic_adar")
    rels = projectable_relations(mfg)
    breakdown: Dict[str, Dict[str, np.ndarray]] = {}
    combine_stack: List[np.ndarray] = []
    for et in rels:
        sims = _relation_similarities(mfg, et, ing_local_149)
        breakdown[et[1]] = sims
        combine_stack.append(_zscore_offdiag(sims[combine_metric]))
    if combine_stack:
        combined = np.mean(np.stack(combine_stack, axis=0), axis=0)
    else:
        combined = np.zeros((len(ing_local_149), len(ing_local_149)))
    np.fill_diagonal(combined, -np.inf)
    return combined, breakdown
