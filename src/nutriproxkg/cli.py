"""Command-line entry points for the local code release."""

import argparse
import logging
import dataclasses
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .pipeline import (
    prepare,
    validate,
    build,
    load_graph,
    load_labels,
    write,
    metadata,
    sha,
)
from .training_config import E2EConfig
from .retrieval.load_graphs import MFGraph
from .retrieval.nutrient_gold import build_gold
from .retrieval.metrics import per_query_metrics, aggregate
from .retrieval.inference_ppr import run_ppr
from .retrieval.inference_projection import run_projection
from .retrieval.inference_metapath import run_metapath


def task1(cfg):
    from scipy.sparse import load_npz
    from sklearn.preprocessing import normalize

    gd, e, info = load_graph(cfg, 1)
    foods, idx, Y, targets = load_labels(cfg, e)
    offsets = {}
    n = 0
    for t in sorted(gd.num_nodes_by_type):
        offsets[t] = n
        n += gd.num_nodes_by_type[t]
    mfg = MFGraph(
        "nutriproxkg",
        "NutriProxKG",
        gd.num_nodes_by_type,
        e["ingredients"],
        {et: gd.edge_index_dict[et] for et in gd.forward_edge_types},
        offsets,
        n,
    )
    gold = build_gold(foods, Y, **cfg["gold"])
    results = {}
    for method in cfg["inference"]["methods"]:
        if method in ("vanilla_ppr", "typed_ppr"):
            gi = mfg.ingredient_global(idx)
            scores = run_ppr(mfg, gi, gi, cfg, typed=method == "typed_ppr")
        elif method == "projection":
            scores, _ = run_projection(mfg, idx, cfg)
        elif method == "metapath":
            scores, _ = run_metapath(mfg, idx, cfg)
        elif method == "direct_cosine":
            X = normalize(
                load_npz(Path(cfg["processed"]) / f"profiles_{cfg['track']}.npz")[idx]
            )
            scores = (X @ X.T).toarray()
        else:
            raise ValueError(f"Unknown method {method}")
        np.fill_diagonal(scores, -np.inf)
        pq = per_query_metrics(scores, gold, cfg)
        order = np.argsort(-scores, axis=1, kind="stable")[:, : len(foods) - 1]
        results[method] = {"mean": aggregate(pq), "per_query": pq, "rankings": order}
    return {
        "task": "task1",
        "track": cfg["track"],
        "foods": foods,
        "targets": targets,
        "evaluation_scope": "user-supplied intersection"
        if cfg.get("foods")
        else "all complete-case foods (not paper common set)",
        "results": results,
    }


def task2(cfg):
    import torch
    from .training import train_predict_general
    from .evaluation import kfold_regression, aggregate_seed_blocks
    from .cv import make_kfold
    from .seed import set_seed

    torch.set_num_threads(cfg["training"].get("threads", 1))
    gd, e, info = load_graph(cfg, 2)
    foods, idx, Y, targets = load_labels(cfg, e)
    training = cfg["training"]
    accepted = {f.name for f in dataclasses.fields(E2EConfig)}
    kwargs = {k: v for k, v in training.items() if k in accepted}
    config = E2EConfig(**kwargs)
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable. Set training.device: cpu or install the appropriate CUDA PyTorch wheel."
        )
    if training["model"] not in ("rgcn_l3", "train_mean"):
        raise ValueError("Supported Task 2 models: rgcn_l3, train_mean")
    blocks = []
    evidence = []
    for seed in training["seeds"]:
        c = dataclasses.replace(config, seed=seed)
        folds = []

        def predict(fold, tr, te, Ystd):
            folds.append(
                {
                    "fold": fold,
                    "train_rows": tr.tolist(),
                    "test_rows": te.tolist(),
                    "y_train_mean": Y[tr].mean(axis=0).tolist(),
                    "y_train_scale": Y[tr].std(axis=0).tolist(),
                }
            )
            if training["model"] == "train_mean":
                return np.zeros((len(te), len(targets)), np.float32)
            return train_predict_general(
                gd, "rgcn_l3", idx, tr, te, Ystd, len(targets), c, fold
            )

        block = kfold_regression(
            predict, Y, targets, c.k_folds, seed, cfg["tolerance"], collect_preds=True
        )
        pred = block.pop("_pred_all")
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite out-of-fold predictions; evaluation aborted")
        blocks.append(block)
        evidence.append(
            {
                "seed": seed,
                "folds": folds,
                "prediction": pred,
                "reference": Y,
                "scores": block,
            }
        )
        print(f"Completed Task 2 seed {seed}: {len(folds)} folds", flush=True)
    return {
        "task": "task2",
        "track": cfg["track"],
        "model": training["model"],
        "foods": foods,
        "targets": targets,
        "evaluation_scope": "user-supplied intersection"
        if cfg.get("foods")
        else "all complete-case foods (not paper common set)",
        "result": aggregate_seed_blocks(blocks, training["seeds"], targets),
        "per_seed": evidence,
        "device": str(config.device),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name()
        if str(config.device).startswith("cuda")
        else None,
    }


def summarize(directory):
    directory = Path(directory)
    rows = []
    for p in sorted(directory.rglob("result.json")):
        obj = json.loads(p.read_text())
        if obj["task"] == "task1":
            for method, v in obj["results"].items():
                rows.append(
                    {
                        "path": str(p.relative_to(directory)),
                        "task": "task1",
                        "track": obj["track"],
                        "method": method,
                        **v["mean"],
                    }
                )
        else:
            rows.append(
                {
                    "path": str(p.relative_to(directory)),
                    "task": "task2",
                    "track": obj["track"],
                    "method": obj["model"],
                    **{
                        k: v
                        for k, v in obj["result"].items()
                        if isinstance(v, (float, int))
                    },
                }
            )
    if not rows:
        raise ValueError(
            f"No result.json files under {directory}; run task1 or task2 first."
        )
    pd.DataFrame(rows).to_csv(directory / "summary.csv", index=False)
    write(directory / "summary.json", rows)
    print(directory / "summary.csv")


def configuration(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    # All relative data paths are relative to the repository (parent of configs).
    root = path.parent.parent
    for key in ["raw", "processed", "output", "tolerance", "foods"]:
        if cfg.get(key):
            cfg[key] = str((root / cfg[key]).resolve())
    if not Path(cfg["tolerance"]).is_file():
        raise ValueError(f"Missing tolerance config: {cfg['tolerance']}")
    if cfg["track"] not in ("n3", "full"):
        raise ValueError("track must be n3 or full")
    g = cfg["graph"]
    t = cfg["training"]
    if g["variant"] not in ("proposed", "ics") or g["rank"] not in (
        "genus",
        "species",
        "family",
    ):
        raise ValueError("Invalid graph variant or rank")
    if (
        not 0 < g["cosine_threshold"] <= 1
        or g["upper_levels"] < 0
        or g["search_limit"] < 1
        or g["min_compounds"] < 1
        or g.get("ics_lineage_limit", 50) < 1
    ):
        raise ValueError("Invalid graph construction parameters")
    if (
        t["k_folds"] < 2
        or t["epochs"] < 1
        or len(t["seeds"]) != len(set(t["seeds"]))
        or not t["seeds"]
    ):
        raise ValueError("Need positive epochs, >=2 folds and distinct seeds")
    if t["model"] == "rgcn_l3" and t.get("n_ii_layers", 3) != 3:
        raise ValueError("rgcn_l3 fixes the encoder depth to three layers")
    if t.get("aux_structural_weight", 0) != 0:
        raise ValueError(
            "This release supports only the paper R-GCN with auxiliary loss disabled"
        )
    if not 0 < t["inner_val_frac"] < 1 or t["topk_compounds"] < 0:
        raise ValueError("Invalid training parameters")
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["validate", "preprocess", "build", "task1", "task2"]:
        q = sub.add_parser(
            name,
            help={
                "validate": "Check required source files and schema",
                "preprocess": "Extract nutrients and construct source arrays",
                "build": "Condense taxonomy and recompute masked similarity",
                "task1": "Learning-free nutritional-similarity retrieval",
                "task2": "Train, infer, and score label-inductive R-GCN or train-fold mean",
            }[name],
        )
        q.add_argument("--config", default="configs/nutri19.yaml")
    q = sub.add_parser("aggregate", help="Collect result.json files into CSV/JSON")
    q.add_argument("--outputs", default="outputs")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "aggregate":
            summarize(args.outputs)
            return
        cfg = configuration(args.config)
        start = time.perf_counter()
        if args.command == "validate":
            print(f"Validated {len(validate(cfg['raw']))} source files")
            return
        if args.command == "preprocess":
            print(prepare(cfg))
            return
        if args.command == "build":
            build(cfg)
            print("Built masked graph for", cfg["track"])
            return
        dest = Path(cfg["output"]) / cfg["track"] / args.command
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "result.json").exists():
            raise ValueError(
                f"{dest}/result.json exists. Choose a new output in the config to preserve earlier runs."
            )
        result = task1(cfg) if args.command == "task1" else task2(cfg)
        inputs = [
            Path(cfg["processed"]) / n
            for n in [
                f"graph_{cfg['track']}.npz",
                f"graph_{cfg['track']}.json",
                f"labels_{cfg['track']}.csv",
                "entities.json",
                "preprocess.json",
                f"profiles_{cfg['track']}.npz",
            ]
        ]
        if cfg.get("foods"):
            inputs.append(Path(cfg["foods"]))
        inputs.append(Path(cfg["tolerance"]))
        result["metadata"] = metadata(cfg, inputs)
        result["elapsed_seconds"] = time.perf_counter() - start
        write(dest / "result.json", result)
        write(dest / "resolved_config.json", cfg)
        print(dest / "result.json")
    except (ValueError, FileNotFoundError, KeyError, yaml.YAMLError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
