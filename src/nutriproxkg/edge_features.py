"""INGREDIENT_COMPOUND edge-feature construction.

FooDB ``FOODB.standard_content`` is missing for ~85% of edges and uses ~30
different units when present.  We build a 3-channel edge feature that is robust
to this mess:

    [ presence , has_quantity , z(log1p(content)) ]

  - presence      : always 1 (the ingredient contains the compound)
  - has_quantity  : 1 iff a comparable mg/100g-family quantity is available
  - z_log_content : standardised log1p(content); 0 when has_quantity == 0

Only the dominant, mutually-comparable ``mg/100g`` family is treated as a real
quantity; all other units (uM, IU, RE, NE, kcal/100g, dry/fresh-weight variants,
...) are kept as presence-only so we never mix incompatible scales into one
scalar.  The z-score uses the **global** distribution of log1p(content) over the
quantity edges: this normalises an *input* feature (compound amounts) and never
touches the nutrition targets, so it introduces no label leakage under the
transductive protocol.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Units that are directly comparable as "mg per 100 g of edible portion".
# Deliberately excludes dry/fresh-weight and per-litre/IU/uM variants.
MG_PER_100G_UNITS = {"mg/100g", "mg/100 g"}


def parse_content(raw: object, unit: object) -> Optional[float]:
    """Return a positive mg/100g content value, or None if not usable."""
    if str(unit).strip() not in MG_PER_100G_UNITS:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0.0:
        return None
    return val


def build_edge_features(contents: np.ndarray, has_quantity: np.ndarray) -> np.ndarray:
    """Build the (E, 3) edge-feature matrix.

    Parameters
    ----------
    contents      : (E,) float32, raw mg/100g content (0 where has_quantity == 0)
    has_quantity  : (E,) bool/float, 1 where a comparable quantity is present
    """
    contents = np.asarray(contents, dtype=np.float64)
    hq = np.asarray(has_quantity, dtype=bool)

    log_c = np.zeros_like(contents)
    log_c[hq] = np.log1p(contents[hq])

    if hq.any():
        mu = float(log_c[hq].mean())
        sigma = float(log_c[hq].std()) + 1e-8
    else:
        mu, sigma = 0.0, 1.0

    z = np.zeros_like(contents)
    z[hq] = (log_c[hq] - mu) / sigma

    feat = np.stack(
        [
            np.ones_like(contents),  # presence
            hq.astype(np.float64),  # has_quantity
            z,  # z(log1p(content))
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def topk_compounds_per_ingredient(
    ing_idx: np.ndarray,
    cmp_idx: np.ndarray,
    feat: np.ndarray,
    k: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap each ingredient to its top-``k`` compounds.

    Priority: measured-quantity edges first (highest content), then a random
    sample of presence-only edges.  Keeps the most informative chemistry while
    bounding the IC edge count for full-batch GPU training.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(ing_idx, kind="stable")
    ing_s = ing_idx[order]
    keep_mask = np.zeros(ing_idx.shape[0], dtype=bool)

    # score: has_quantity gives a large base so measured edges sort first,
    # then by z(log content); presence-only edges get a small random jitter.
    has_q = feat[:, 1]
    z = feat[:, 2]
    score = has_q * 1e6 + z + rng.random(ing_idx.shape[0]) * 1e-3
    score_s = score[order]

    boundaries = np.flatnonzero(np.diff(ing_s)) + 1
    groups = np.split(np.arange(ing_s.shape[0]), boundaries)
    for grp in groups:
        if grp.size <= k:
            keep_mask[order[grp]] = True
            continue
        local = grp[np.argsort(score_s[grp])[::-1][:k]]
        keep_mask[order[local]] = True

    return ing_idx[keep_mask], cmp_idx[keep_mask], feat[keep_mask]
