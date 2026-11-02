"""Source-data adapters and reproducible graph construction.

The loaders, edge features, masking and cosine routine come from the research
implementation. This module connects them without requiring cached research data.
"""

from pathlib import Path
import csv
import hashlib
import json
import platform
import importlib.metadata
import logging
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from .loaders.foodb_loader import FooDBLoader
from .loaders.foodon_loader import FoodOnLoader
from .loaders.base import iter_table_rows, parse_float
from .edge_features import (
    parse_content,
    build_edge_features,
    topk_compounds_per_ingredient,
)
from .nutrient_aliases import masked_compound_ids, targets_for_track
from .similarity import cosine_edges
from .encoder import GraphData

logger = logging.getLogger(__name__)

REQUIRED = {
    "foodb/Food.csv": ["id", "name", "ncbi_taxonomy_id"],
    "foodb/Compound.csv": ["id", "name"],
    "foodb/Nutrient.csv": ["id", "name"],
    "foodb/Content.csv": [
        "food_id",
        "source_id",
        "source_type",
        "standard_content",
        "orig_unit",
    ],
}


def norm(s):
    return " ".join(str(s).split()).lower()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    return x


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def metadata(cfg, inputs):
    versions = {}
    for name in [
        "nutriproxkg",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "rdflib",
        "PyYAML",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "config": cfg,
        "python": platform.python_version(),
        "platform": platform.system(),
        "versions": versions,
        "inputs": {
            str(p): {"sha256": sha(p), "bytes": Path(p).stat().st_size} for p in inputs
        },
        "code_sha256": {
            str(p.relative_to(Path(__file__).parent)): sha(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
    }


def validate(raw):
    raw = Path(raw)
    files = []
    for name, fields in REQUIRED.items():
        p = raw / name
        if not p.is_file():
            raise ValueError(
                f"Missing {p}. Download and extract the source as described in docs/data.md."
            )
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = set(fields) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{p}: missing columns {sorted(missing)}")
            if next(reader, None) is None:
                raise ValueError(f"{p}: empty table")
        files.append(p)
    p = raw / "foodon/foodon.owl"
    if not p.is_file():
        raise ValueError(f"Missing {p}: provide FoodOn RDF/XML OWL; see docs/data.md.")
    files.append(p)
    for name, min_fields in [("nodes.dmp", 3), ("names.dmp", 4), ("merged.dmp", 2)]:
        p = raw / "ncbi" / name
        if not p.is_file():
            raise ValueError(f"Missing {p}: extract NCBI taxdump.tar.gz here.")
        with p.open() as f:
            first = f.readline()
        if first and len(first.split("|")) < min_fields:
            raise ValueError(f"{p}: malformed NCBI pipe-separated record")
        if not first and name != "merged.dmp":
            raise ValueError(f"{p}: empty taxonomy table")
        files.append(p)
    return files


def legacy_taxonomy_identity(parent, rank, names_path):
    """Project NCBI maps through the original builder's scientific-name merging.

    MultiTypeGraphBuilder merges Species by normalized name and aggregates differing
    scalar attributes as sorted string lists. Reproduce that representation here
    without materializing the full multi-million-node NetworkX graph. Raw taxon-ID
    identity would change the historical genus anchors (see reproducibility.md).
    Parent attribute values deliberately retain the original builder's semantics.
    """
    scientific_names = {}
    with Path(names_path).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = [value.strip() for value in line.split("|")]
            if len(fields) >= 4 and fields[3] == "scientific name":
                scientific_names[fields[0]] = norm(fields[1])
    first_id = {}
    groups = {}
    for tid in parent:
        name = scientific_names.get(tid, "taxon_" + tid)
        if name in first_id:
            groups.setdefault(first_id[name], [first_id[name]]).append(tid)
        else:
            first_id[name] = tid

    def merge(values):
        values = sorted(set(value for value in values if value))
        return values[0] if len(values) == 1 else repr(values)

    remap = {}
    for ids in groups.values():
        key = merge(ids)
        parent_value = merge(parent[tid] for tid in ids)
        rank_value = merge(rank[tid] for tid in ids)
        for tid in ids:
            remap[tid] = key
            del parent[tid]
            del rank[tid]
        parent[key] = parent_value
        rank[key] = rank_value
    return remap, len(groups)


def prepare(cfg):
    raw = Path(cfg["raw"])
    dest = Path(cfg["processed"])
    files = validate(raw)
    dest.mkdir(parents=True, exist_ok=True)
    loader = FooDBLoader(raw / "foodb")
    foods, links = loader._load_foods(raw / "foodb/Food.csv", None)
    compounds = loader._load_compounds(raw / "foodb/Compound.csv")
    nutrients, _ = loader._load_nutrients(raw / "foodb/Nutrient.csv")
    compounds.update(nutrients)
    # Exact case/whitespace name merging, also used by MultiTypeGraphBuilder.
    names = {norm(v.name): v.name for v in reversed(list(foods.values()))}
    logger.info("Parsing FoodOn hierarchy")
    onodes, oedges = FoodOnLoader(raw / "foodon")._load_from_owl(
        raw / "foodon/foodon.owl", max_edges=None
    )
    if not oedges:
        raise ValueError(
            "FoodOn has no labelled FOODON-to-FOODON subClassOf edges. Check foodon.owl."
        )
    for v in onodes.values():
        names.setdefault(norm(v["name"]), v["name"])
    keys = sorted(names)
    ii = {k: i for i, k in enumerate(keys)}
    cnames = {}
    for v in compounds.values():
        cnames.setdefault(norm(v.name), v.name)
    ckeys = sorted(cnames)
    ci = {k: i for i, k in enumerate(ckeys)}
    foodidx = {k: ii[norm(v.name)] for k, v in foods.items()}
    compidx = {k: ci[norm(v.name)] for k, v in compounds.items()}
    isa = []
    for a, b, _ in oedges:
        isa.append((ii[norm(onodes[a]["name"])], ii[norm(onodes[b]["name"])]))
    hc = []
    amounts = []
    positive = []
    content_counts = Counter()
    unit_counts = Counter()
    logger.info("Streaming FooDB content records")
    for row in iter_table_rows(raw / "foodb/Content.csv"):
        content_counts["total"] += 1
        if content_counts["total"] % 1000000 == 0:
            logger.info("Read %d content records", content_counts["total"])
        unit_counts[row.get("orig_unit", "")] += 1
        fid = row.get("food_id")
        st = row.get("source_type", "").lower()
        sid = row.get("source_id", "")
        ck = (
            ("Compound:" + sid)
            if st == "compound"
            else ("Nutrient:" + sid if st == "nutrient" else "")
        )
        if fid not in foodidx or ck not in compidx:
            content_counts["unmatched_join"] += 1
            continue
        a, b = foodidx[fid], compidx[ck]
        hc.append((a, b))
        val = parse_content(row.get("standard_content"), row.get("orig_unit"))
        amounts.append(val)
        pos = parse_float(row.get("standard_content"))
        if pos is not None and np.isfinite(pos) and pos > 0:
            positive.append((a, b))
    if not hc:
        raise ValueError(
            "Content joins produced no ingredient-compound edges. Check source_type and internal identifiers."
        )
    logger.info("Reading NCBI parent and rank maps")
    parent = {}
    rank = {}
    with (raw / "ncbi/nodes.dmp").open() as f:
        for n, line in enumerate(f, 1):
            p = [s.strip() for s in line.split("|")]
            if len(p) < 3 or not p[0] or not p[1]:
                raise ValueError(f"nodes.dmp line {n}: invalid record")
            parent[p[0]] = p[1]
            rank[p[0]] = p[2]
    logger.info("Applying legacy scientific-name identity to taxonomy")
    taxonomy_remap, taxonomy_name_collisions = legacy_taxonomy_identity(
        parent, rank, raw / "ncbi/names.dmp"
    )
    merged = {}
    with (raw / "ncbi/merged.dmp").open() as f:
        for line in f:
            p = [s.strip() for s in line.split("|")]
            if len(p) >= 2:
                merged[p[0]] = p[1]
    anchors = {}
    for link in links:
        tid = merged.get(link.tax_id, link.tax_id)
        tid = taxonomy_remap.get(tid, tid)
        idx = foodidx[link.food_id]
        if idx in anchors and anchors[idx] != tid:
            raise ValueError(
                f"Conflicting NCBI anchors for merged food {names[keys[idx]]}"
            )
        if tid in parent:
            anchors[idx] = tid
    hc = edges(hc)
    pos = edges(positive)
    X = csr_matrix(
        (np.ones(pos.shape[1], dtype=np.float32), (pos[0], pos[1])),
        shape=(len(keys), len(ckeys)),
    )
    X.sum_duplicates()
    X.data[:] = 1
    vals = np.array([v if v is not None else 0 for v in amounts])
    feat = build_edge_features(vals, np.array([v is not None for v in amounts]))
    np.savez_compressed(
        dest / "base.npz", has_compound=hc, ic_feat=feat, is_a=edges(isa)
    )
    save_npz(dest / "profiles.npz", X)
    write(
        dest / "entities.json",
        {
            "ingredients": [names[k] for k in keys],
            "compounds": [cnames[k] for k in ckeys],
            "anchors": anchors,
            "parent": parent,
            "rank": rank,
        },
    )
    # Preserve extraction semantics, including the legacy unit conversion; see audit.
    logger.info("Extracting complete-case nutrient targets")
    df = pd.DataFrame(
        loader.extract_nutrition_table(
            include_micronutrients=True, estimate_water=False
        )
    )
    if df.empty:
        raise ValueError(
            "No foods with complete macronutrients; check FooDB units and nutrient names."
        )
    df = df.drop_duplicates(subset=["ingredient_name"])
    dropped = []
    for col in [c for c in df if c != "ingredient_name"]:
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty or (finite.eq(0).mean() >= 0.5):
            dropped.append(col)
    for track in ["n3", "full"]:
        targets = targets_for_track(track)
        if track == "full" and set(targets) & set(dropped):
            raise ValueError(
                f"Nutri-19 target unexpectedly sparse: {set(targets) & set(dropped)}. Source version differs; do not silently change the targets."
            )
        table = (
            df[["ingredient_name"] + targets]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if table.empty:
            raise ValueError(f"No complete cases for {track}")
        table.to_csv(dest / f"labels_{track}.csv", index=False)
    write(
        dest / "preprocess.json",
        {
            "metadata": metadata(cfg, files),
            "dropped_nutrients": dropped,
            "ingredient_order": "sorted normalized names",
            "counts": {
                "ingredients": len(keys),
                "compounds": len(ckeys),
                "content_records": hc.shape[1],
                "content_audit": dict(content_counts),
                "taxonomy_name_collision_groups": taxonomy_name_collisions,
                "orig_units": dict(unit_counts),
                "complete_cases": {
                    t: len(pd.read_csv(dest / f"labels_{t}.csv"))
                    for t in ["n3", "full"]
                },
            },
        },
    )
    return dest


def edges(pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy()


def taxonomy(entities, rank_name, levels, search_limit, lineage_limit=50):
    parent = entities["parent"]
    ranks = entities["rank"]
    anchors = {}
    for idx, tid in entities["anchors"].items():
        seen = set()
        for _ in range(search_limit):
            if tid not in ranks or tid in seen:
                break
            seen.add(tid)
            if rank_name == "original" or ranks[tid] == rank_name:
                anchors[int(idx)] = tid
                break
            tid = parent.get(tid, "")
    kept = set(anchors.values())
    front = set(kept)
    for _ in range(levels if rank_name != "original" else lineage_limit - 1):
        front = {
            parent[t] for t in front if parent.get(t) in parent and parent[t] != t
        } - kept
        if not front:
            break
        kept.update(front)
    tids = sorted(kept)
    lookup = {t: i for i, t in enumerate(tids)}
    return (
        tids,
        edges([(i, lookup[t]) for i, t in sorted(anchors.items())]),
        edges(
            [
                (lookup[t], lookup[parent[t]])
                for t in tids
                if parent.get(t) in lookup and parent[t] != t
            ]
        ),
    )


def build(cfg):
    dest = Path(cfg["processed"])
    required = [
        dest / n
        for n in ["base.npz", "profiles.npz", "entities.json", "preprocess.json"]
    ]
    for p in required:
        if not p.is_file():
            raise ValueError(f"Missing {p}; run preprocess first.")
    e = json.loads((dest / "entities.json").read_text())
    base = np.load(dest / "base.npz")
    X = load_npz(dest / "profiles.npz")
    eligible = (
        np.asarray(X.sum(axis=1)).ravel() >= cfg["graph"]["min_compounds"]
    )  # legacy pre-mask eligibility
    masked = masked_compound_ids(e["compounds"], cfg["track"])
    keep = np.ones(X.shape[1], dtype=np.float32)
    keep[list(masked)] = 0
    X = X.multiply(keep).tocsr()
    X.eliminate_zeros()
    graph = cfg["graph"]
    tids, fg, ss = taxonomy(
        e,
        "original" if graph["variant"] == "ics" else graph["rank"],
        graph["upper_levels"],
        graph["search_limit"],
        graph.get("ics_lineage_limit", 50),
    )
    sim = (
        edges([])
        if graph["variant"] == "ics"
        else cosine_edges(X, eligible, graph["cosine_threshold"])
    )
    # Store all containment records; cap is applied before masking for Task 2,
    # matching e2e.load_graph_pair -> apply_nutrient_mask in the research code.
    np.savez_compressed(
        dest / f"graph_{cfg['track']}.npz",
        has_compound=base["has_compound"],
        ic_feat=base["ic_feat"],
        is_a=base["is_a"],
        from_genus=fg,
        subtaxon_of=ss,
        similar_to=sim,
    )
    write(
        dest / f"graph_{cfg['track']}.json",
        {
            "species": tids,
            "masked_compounds": sorted(masked),
            "graph_config": graph,
            "metadata": metadata(cfg, required),
        },
    )
    save_npz(dest / f"profiles_{cfg['track']}.npz", X)


def load_graph(cfg, task):
    dest = Path(cfg["processed"])
    tag = cfg["track"]
    p = dest / f"graph_{tag}.npz"
    if not p.is_file():
        raise ValueError(f"Missing {p}; run build first.")
    e = json.loads((dest / "entities.json").read_text())
    info = json.loads((dest / f"graph_{tag}.json").read_text())
    for path, record in info["metadata"]["inputs"].items():
        if not Path(path).is_file() or sha(path) != record["sha256"]:
            raise ValueError("Preprocessed inputs changed or moved; run build again")
    if info["graph_config"] != cfg["graph"]:
        raise ValueError(
            "Graph configuration changed; run build again before evaluation."
        )
    raw = np.load(p)
    hc = raw["has_compound"]
    feat = raw["ic_feat"]
    if task == 2 and cfg["training"]["topk_compounds"]:
        a, b, feat = topk_compounds_per_ingredient(
            hc[0],
            hc[1],
            feat,
            cfg["training"]["topk_compounds"],
            seed=cfg["training"]["graph_seed"],
        )
        hc = np.asarray([a, b])
    mask = info["masked_compounds"]
    hc = hc[:, ~np.isin(hc[1], mask)]
    types = ["Ingredient", "Compound", "Species"]
    sizes = {
        t: len(e[k]) for t, k in zip(types, ["ingredients", "compounds", "ingredients"])
    }
    sizes["Species"] = len(info["species"])
    rels = {}
    for s, r, d, arr in [
        ("Ingredient", "has_compound", "Compound", hc),
        ("Ingredient", "from_genus", "Species", raw["from_genus"]),
        ("Species", "subtaxon_of", "Species", raw["subtaxon_of"]),
        ("Ingredient", "is_a", "Ingredient", raw["is_a"]),
        ("Ingredient", "similar_to", "Ingredient", raw["similar_to"]),
    ]:
        if r in cfg.get("drop_relations", []) or arr.shape[1] == 0:
            continue
        rels[s, r, d] = np.ascontiguousarray(arr)
    forward = list(rels)
    for s, r, d in forward:
        rels[d, "rev_" + r, s] = rels[s, r, d][::-1].copy()
    gd = GraphData(
        {
            "Ingredient": e["ingredients"],
            "Compound": e["compounds"],
            "Species": info["species"],
        },
        sizes,
        rels,
        forward,
        (types, list(rels)),
        types,
    )
    return gd, e, info


def load_labels(cfg, e):
    dest = Path(cfg["processed"])
    df = pd.read_csv(dest / f"labels_{cfg['track']}.csv")
    targets = targets_for_track(cfg["track"])
    scope = cfg.get("foods")
    if scope:
        record = json.loads(Path(scope).read_text())
        foods = record["common_foods"] if isinstance(record, dict) else record
        if len(foods) != len(set(foods)):
            raise ValueError("Evaluation food list contains duplicates")
        missing = set(foods) - set(df.ingredient_name)
        if missing:
            raise ValueError(
                f"Evaluation food list has {len(missing)} foods without complete labels"
            )
        # Preserve CSV order, matching historical label alignment.
        df = df[df.ingredient_name.isin(foods)]
    lut = {norm(n): i for i, n in enumerate(e["ingredients"])}
    foods = df.ingredient_name.tolist()
    idx = np.array([lut[norm(n)] for n in foods], dtype=np.int64)
    if len(set(idx)) != len(idx):
        raise ValueError(
            "Duplicate ingredient identity after normalization in evaluation labels"
        )
    Y = df[targets].to_numpy(dtype=np.float32)
    if (
        len(foods) < max(cfg["training"]["k_folds"], 4)
        or not np.isfinite(Y).all()
        or np.any(Y < 0)
    ):
        raise ValueError(
            "Need enough complete, finite, nonnegative nutrient rows for the configured folds"
        )
    return foods, idx, Y, targets
