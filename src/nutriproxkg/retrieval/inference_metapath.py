"""Metapath / PathSim ingredient similarity (model-free, no learning).

Each graph contributes one symmetric I–X–I metapath per relation that touches
Ingredient:

  FlavorGraph : I-Compound-I, I-is_a-I
  FoodKG      : I-Compound-I, I-Recipe-I
  ICS         : I-Compound-I, I-genus(Species)-I, I-is_a-I
  ChemTaxFoodKG : I-Compound-I, I-genus(Species)-I, I-similar_to-I, I-is_a-I

Note: in the loaded ICS_typed / ChemTaxFoodKG_full tensors the taxonomy edge is
Ingredient→Species (``from_genus``) and the chemical-similarity edge is
Ingredient↔Ingredient (``similar_to``), so the design-doc metapaths
``I-Compound-Species-Compound-I`` and ``I-Compound-similar_to-Compound-I`` are
realized as the equivalent direct 2-hop metapaths I-genus-I and I-similar_to-I
(see METHODS.md).

PathSim(i,j) = 2·M[i,j] / (M[i,i] + M[j,j]),  M = commuting matrix of the
metapath (count of length-2 paths).  The combined ranking is the equal-weight
z-mean of the per-metapath PathSim matrices (no learned weights).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .graph_preprocess import ingredient_incidence, projectable_relations
from .inference_projection import _zscore_offdiag
from .load_graphs import MFGraph

_METAPATH_LABEL = {
    "has_compound": "I-Compound-I",
    "is_a": "I-is_a-I",
    "has_ingredient": "I-Recipe-I",
    "from_genus": "I-genus-I",
    "similar_to": "I-similar_to-I",
}


def metapaths_for(mfg: MFGraph) -> List[Tuple[str, tuple]]:
    """(label, relation_edge_type) for each I–X–I metapath available in the graph."""
    out = []
    for et in projectable_relations(mfg):
        out.append((_METAPATH_LABEL.get(et[1], f"I-{et[1]}-I"), et))
    return out


def _pathsim(mfg: MFGraph, et, ing_local_149: np.ndarray) -> np.ndarray:
    B, _, _ = ingredient_incidence(mfg, et)
    B149 = B[ing_local_149]
    M = (B149 @ B149.T).toarray().astype(np.float64)  # commuting matrix
    diag = np.diag(M).copy()
    denom = diag[:, None] + diag[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = np.where(denom > 0, 2.0 * M / denom, 0.0)
    np.fill_diagonal(ps, -np.inf)
    return ps


def run_metapath(
    mfg: MFGraph, ing_local_149: np.ndarray, cfg: dict
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (combined PathSim 149x149, per-metapath PathSim breakdown)."""
    mps = metapaths_for(mfg)
    per: Dict[str, np.ndarray] = {}
    stack: List[np.ndarray] = []
    for label, et in mps:
        ps = _pathsim(mfg, et, ing_local_149)
        per[label] = ps
        stack.append(_zscore_offdiag(ps))
    if stack:
        combined = np.mean(np.stack(stack, axis=0), axis=0)  # equal-weight z-mean
    else:
        combined = np.zeros((len(ing_local_149), len(ing_local_149)))
    np.fill_diagonal(combined, -np.inf)
    return combined, per
