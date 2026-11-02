"""Sparse structural views of an MFGraph (no nutrient signal, no learning).

Provides:
  - untyped_adjacency  : binary global adjacency (optionally undirected)
  - row_normalize      : row-stochastic transition P
  - typed_transition   : typed-normalized transition (equal mass per relation,
                         then equal within the relation) — the main PPR operator
  - ingredient_incidence: ingredient -> entity binary incidence per relation
                          (entity = the non-ingredient endpoint, or the other
                          ingredient for I-I relations), plus global entity degree
  - ego_subgraph       : per-query k-hop ego edge sets with relation-wise caps
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .load_graphs import MFGraph, REL


def _global_edges(
    mfg: MFGraph, relations: Optional[List[REL]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenated global (src, dst) over the selected forward relations."""
    rels = relations if relations is not None else list(mfg.relations.keys())
    src_parts, dst_parts = [], []
    for et in rels:
        ei = mfg.relations.get(et)
        if ei is None or ei.shape[1] == 0:
            continue
        s, _, d = et
        src_parts.append(ei[0] + mfg.type_offset[s])
        dst_parts.append(ei[1] + mfg.type_offset[d])
    if not src_parts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(src_parts), np.concatenate(dst_parts)


def untyped_adjacency(
    mfg: MFGraph, relations: Optional[List[REL]] = None, directed: bool = False
) -> sp.csr_matrix:
    """Binary global adjacency.  Undirected (default) symmetrises and re-binarises."""
    s, d = _global_edges(mfg, relations)
    n = mfg.n_total
    A = sp.csr_matrix((np.ones(s.size), (s, d)), shape=(n, n))
    if not directed:
        A = A + A.T
    A = A.tocsr()
    A.data[:] = 1.0  # binarise (ignore multiplicity / quantity)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    """Row-stochastic transition P (rows with no out-edges stay all-zero)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.diags(inv) @ A


def typed_transition(
    mfg: MFGraph, relations: Optional[List[REL]] = None, directed: bool = False
) -> sp.csr_matrix:
    """Typed-normalized transition matrix.

    For each node u: split outgoing probability EQUALLY across the relation
    types present at u, then EQUALLY across the neighbours within that relation.
    This prevents a high-degree relation (e.g. has_compound) or a huge node type
    from dominating the walk.

        P = diag(1/R_u) @ sum_r rownorm(A_r)

    where A_r is the (undirected) adjacency of relation r and R_u is the number
    of relations in which u has at least one neighbour.
    """
    rels = relations if relations is not None else list(mfg.relations.keys())
    n = mfg.n_total
    P_sum = sp.csr_matrix((n, n))
    rel_count = np.zeros(n)
    for et in rels:
        A_r = untyped_adjacency(mfg, [et], directed=directed)
        if A_r.nnz == 0:
            continue
        P_r = row_normalize(A_r)
        P_sum = P_sum + P_r
        has = (np.asarray(A_r.sum(axis=1)).ravel() > 0).astype(np.float64)
        rel_count += has
    inv = np.divide(1.0, rel_count, out=np.zeros_like(rel_count), where=rel_count > 0)
    return (sp.diags(inv) @ P_sum).tocsr()


# ---------------------------------------------------------------------------
# Ingredient projections (for projection / metapath similarity)
# ---------------------------------------------------------------------------
def ingredient_incidence(
    mfg: MFGraph, et: REL
) -> Tuple[sp.csr_matrix, np.ndarray, str]:
    """Binary ingredient -> entity incidence for one relation.

    entity = the non-ingredient endpoint; for an I-I relation the entity is the
    *other* ingredient (so two ingredients are "similar" if they share is_a /
    similar_to neighbours).  Returns (B [n_ing x n_entity], entity_degree, entity_type)
    where entity_degree counts how many ingredients touch each entity (for AA/RA).
    """
    s, rel, d = et
    ei = mfg.relations[et]
    n_ing = mfg.num_nodes_by_type["Ingredient"]
    if s == "Ingredient" and d == "Ingredient":
        # symmetric I-I: ingredient i -> ingredient-neighbours (both directions)
        rows = np.concatenate([ei[0], ei[1]])
        cols = np.concatenate([ei[1], ei[0]])
        n_ent = n_ing
        ent_type = "Ingredient"
    elif s == "Ingredient":
        rows, cols = ei[0], ei[1]
        n_ent = mfg.num_nodes_by_type[d]
        ent_type = d
    elif d == "Ingredient":
        rows, cols = ei[1], ei[0]
        n_ent = mfg.num_nodes_by_type[s]
        ent_type = s
    else:
        raise ValueError(f"Relation {et} does not touch Ingredient; cannot project.")
    B = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n_ing, n_ent))
    B.data[:] = 1.0
    B = B.tocsr()
    B.sum_duplicates()
    B.data[:] = 1.0
    entity_degree = np.asarray(B.sum(axis=0)).ravel()  # #ingredients per entity
    return B, entity_degree, ent_type


def projectable_relations(mfg: MFGraph) -> List[REL]:
    """Relations that touch Ingredient (i.e. can be projected to I-I similarity)."""
    return [et for et in mfg.relations if "Ingredient" in (et[0], et[2])]


# ---------------------------------------------------------------------------
# Budget-matched ego subgraphs
# ---------------------------------------------------------------------------
def ego_edges(
    mfg: MFGraph,
    source_local_ing: int,
    *,
    max_hops: int,
    max_neighbors_per_relation: int,
    max_nodes: int,
    max_edges: int,
    rng: np.random.Generator,
    sampling: str = "relation_uniform",
) -> Dict[REL, np.ndarray]:
    """Extract a capped k-hop ego edge set around one ingredient (global ids).

    BFS expands hop-by-hop.  At each frontier node, for each relation we keep at
    most ``max_neighbors_per_relation`` neighbours (uniform random sample with
    ``rng``; relation-uniform expansion).  Global ``max_nodes`` / ``max_edges``
    caps are enforced.  Returns per-relation global edge arrays (2, E).
    """
    src_g = source_local_ing + mfg.type_offset["Ingredient"]
    adj = _relation_csr(mfg)  # per-relation symmetric CSR, cached

    visited = {src_g}
    frontier = [src_g]
    kept: Dict[REL, List[Tuple[int, int]]] = {et: [] for et in mfg.relations}
    n_edges = 0
    for _hop in range(max_hops):
        nxt = []
        for u in frontier:
            for et in mfg.relations:
                A = adj[et]
                neigh = A.indices[A.indptr[u] : A.indptr[u + 1]]
                if neigh.size == 0:
                    continue
                if sampling == "degree":
                    pick = neigh[:max_neighbors_per_relation]
                else:  # relation_uniform
                    if neigh.size > max_neighbors_per_relation:
                        pick = rng.choice(
                            neigh, size=max_neighbors_per_relation, replace=False
                        )
                    else:
                        pick = neigh
                for v in pick:
                    if n_edges >= max_edges:
                        break
                    v = int(v)
                    # node cap is a TRUE cap: skip an edge that would add a NEW
                    # node beyond max_nodes (the ego node set is rebuilt from the
                    # kept edges downstream, so every endpoint must stay in budget).
                    if v not in visited and len(visited) >= max_nodes:
                        continue
                    kept[et].append((u, v))
                    n_edges += 1
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
                if n_edges >= max_edges:
                    break
            if n_edges >= max_edges or len(visited) >= max_nodes:
                break
        frontier = nxt
        if not frontier or n_edges >= max_edges or len(visited) >= max_nodes:
            break

    out: Dict[REL, np.ndarray] = {}
    for et, pairs in kept.items():
        if pairs:
            arr = np.asarray(pairs, dtype=np.int64).T  # (2, E) global ids
            out[et] = arr
    return out


def _relation_csr(mfg: MFGraph) -> Dict[REL, sp.csr_matrix]:
    """Per-relation symmetric (undirected) binary CSR over global ids.

    Cached on the MFGraph INSTANCE (not by id()): a global id()-keyed cache is
    unsafe because Python reuses object ids after garbage collection, so a new
    graph could collide with a freed graph's cached adjacency.
    """
    cache = getattr(mfg, "_rel_csr_cache", None)
    if cache is None:
        cache = {
            et: untyped_adjacency(mfg, [et], directed=False) for et in mfg.relations
        }
        mfg._rel_csr_cache = cache  # type: ignore[attr-defined]
    return cache
