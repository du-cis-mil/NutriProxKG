"""Ranking metrics for nutritional-similarity retrieval.

Every metric is computed PER QUERY (so we can bootstrap over queries) and then
aggregated by mean.  The graph ranking for a query is the candidates sorted by
descending graph score, keeping only REACHABLE candidates (finite score); the
self candidate is always -inf and therefore excluded.

  primary : nDCG@10, Recall@10, Spearman(graph score, gold nutrient similarity)
  aux     : nDCG@{5,20}, Recall@{5,20}, mAP@10, MRR, coverage,
            reachable count, degree correlation
            (+ avg graph distance, supplied separately for the full condition)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .nutrient_gold import Gold

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def _graph_order(score_row: np.ndarray) -> np.ndarray:
    """Reachable candidate indices, sorted by descending score (self is -inf)."""
    finite = np.where(np.isfinite(score_row))[0]
    if finite.size == 0:
        return finite
    return finite[np.argsort(-score_row[finite], kind="stable")]


def _dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    if g.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * discounts))


def ndcg_at_k(order: np.ndarray, relevance_row: np.ndarray, k: int) -> float:
    if order.size == 0:
        return 0.0
    gains = relevance_row[order]
    ideal = np.sort(relevance_row[np.isfinite(relevance_row)])[::-1]
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return _dcg(gains, k) / idcg


def recall_at_k(order: np.ndarray, gold_topk: List[int], k: int) -> float:
    if not gold_topk:
        return np.nan
    topk = set(order[:k].tolist())
    return len(topk & set(gold_topk)) / float(len(gold_topk))


def ap_at_k(order: np.ndarray, positives: set, k: int) -> float:
    if not positives:
        return np.nan
    hits, ap = 0, 0.0
    for i, c in enumerate(order[:k], start=1):
        if c in positives:
            hits += 1
            ap += hits / i
    return ap / min(len(positives), k)


def mrr(order: np.ndarray, positives: set) -> float:
    if not positives:
        return np.nan
    for i, c in enumerate(order, start=1):
        if c in positives:
            return 1.0 / i
    return 0.0


def per_query_metrics(
    scores: np.ndarray,
    gold: Gold,
    cfg: dict,
    candidate_degree: Optional[np.ndarray] = None,
    avg_distance: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute all metrics per query. ``scores`` is (n, n), foods-aligned."""
    n = scores.shape[0]
    mcfg = cfg["metrics"]
    ndcg_ks = list(mcfg["ndcg_k"])
    recall_ks = list(mcfg["recall_k"])
    map_k = int(mcfg["map_k"])

    out: Dict[str, List[float]] = {}
    for k in ndcg_ks:
        out[f"ndcg@{k}"] = []
    for k in recall_ks:
        out[f"recall@{k}"] = []
    out[f"map@{map_k}"] = []
    out["mrr"] = []
    out["spearman"] = []
    out["coverage"] = []
    out["reachable"] = []
    out["degree_corr"] = []

    mrr_pos_k = 10 if 10 in gold.topk else min(gold.topk)
    for q in range(n):
        row = scores[q]
        order = _graph_order(row)
        rel = gold.relevance[q]
        for k in ndcg_ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(order, rel, k))
        for k in recall_ks:
            out[f"recall@{k}"].append(recall_at_k(order, gold.topk[k][q], k))
        out[f"map@{map_k}"].append(ap_at_k(order, set(gold.topk[map_k][q]), map_k))
        out["mrr"].append(mrr(order, set(gold.topk[mrr_pos_k][q])))

        # spearman over reachable candidates (exclude self & non-finite gold)
        finite = np.isfinite(row) & np.isfinite(gold.sim[q])
        finite[q] = False
        if spearmanr is not None and finite.sum() >= 3:
            rho = spearmanr(row[finite], gold.sim[q][finite]).correlation
            out["spearman"].append(float(rho) if rho == rho else np.nan)
        else:
            out["spearman"].append(np.nan)

        reach = int(np.isfinite(row).sum())  # self is -inf -> not counted
        out["reachable"].append(reach)
        out["coverage"].append(reach / float(n - 1))

        if candidate_degree is not None and np.isfinite(row).sum() >= 3:
            f = np.isfinite(row)
            f[q] = False
            d = candidate_degree[f].astype(np.float64)
            s = row[f]
            if spearmanr is not None and d.std() > 0 and s.std() > 0:
                dc = spearmanr(s, d).correlation
                out["degree_corr"].append(float(dc) if dc == dc else np.nan)
            else:
                out["degree_corr"].append(np.nan)
        else:
            out["degree_corr"].append(np.nan)

    res = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    if avg_distance is not None:
        res["avg_distance"] = np.asarray(avg_distance, dtype=np.float64)
    return res


def aggregate(per_query: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for k, v in per_query.items():
        finite = np.isfinite(v)
        out[k] = float(v[finite].mean()) if finite.any() else float("nan")
    return out
